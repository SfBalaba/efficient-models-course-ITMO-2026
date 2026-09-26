def get_kernel_ops_sout(s_in, k, s):
    p = k // 2
    s_out = (s_in + 2 * p - k) // s + 1
    return s_out


NUM_CLASSES = 100
C_IN = 3


def flops(image_size, batch):
    """
    Calculates total FLOPs for one forward pass.
    Convention: 1 MAC = 2 FLOPs.
    """
    S = image_size
    
    B = batch
    total_flops = 0

    def conv_flops(cin, cout, k, s, s_in):
        s_out = get_kernel_ops_sout(s_in, k,  s)
        mac =  cin * k * k * cout * s_out * s_out
        return 2 * mac, cout, s_out

    def relu(cin, s_in):
        return 0, cin, s_in

    def max_pool(cin, k, s, s_in):
        s_out = get_kernel_ops_sout(s_in, k,  s )
        return 0, cin, s_out

    def batch_norm(cin, s_in):
        mac = cin * s_in * s_in
        return 2 * mac, cin, s_in

    def global_avg_pool(cin, s_in):
        # out_shapes = (cin, 1)
        # if mode == 'strict':
        # return cin * s_in * s_in, *(cin, 1)
        # else:
        return 0, *(cin, 1)
    def linear(cin, cout):
        mac = cin  * cout
        return 2 * mac, cout
        

    flops_conv7, cout, s_out = conv_flops(cin=C_IN, cout=32, k=7,  s=2, s_in=S)
    flops_relu1, cout, s_out = relu(cout, s_out)
    
    flops_max_pool, cout, s_out = max_pool(cin=cout, k=3, s=2, s_in=s_out)
  
    flops_conv5, cout, s_out = conv_flops(cin=cout, cout=64, k=5, s=1, s_in=s_out)
    flops_relu2, cout, s_out = relu(cin=cout, s_in=s_out)
    
    flops_conv3, cout, s_out = conv_flops(cin=cout, cout=128, k=3,  s=2, s_in=s_out)
    flops_relu3, cout, s_out = relu(cin=cout, s_in=s_out)

    flops_conv1, cout, s_out = conv_flops(cin=cout, cout=256, k=1, s=1, s_in=s_out)
    flops_relu4, cout, s_out = relu(cin=cout, s_in=s_out)

    flops_conv3_1, cout, s_out = conv_flops(cin=cout, cout=256, k=3, s=2, s_in=s_out)
    flops_relu5, cout, s_out = relu(cin=cout, s_in=s_out)

    flops_conv1_1, cout, s_out = conv_flops(cin=cout, cout=512, k=1, s=1, s_in=s_out)
    flops_relu6, cout, s_out = relu(cin=cout, s_in=s_out)
    
    flops_gap, cout, s_out = global_avg_pool(cin=cout, s_in=s_out)

    flat = cout * s_out
    
    flops_linear1, cout = linear(cin=flat, cout=256)
    
    flops_relu7, cout, s_out = relu(cin=cout, s_in=s_out)
    
    flops_linear2, cout = linear(cin=cout, cout=NUM_CLASSES)
    
    total_flops = B * (flops_conv7 + flops_relu1 \
                       + flops_max_pool \
                       + flops_conv5 + flops_relu2 \
                       + flops_conv3 + flops_relu3 \
                       + flops_conv1 + flops_relu4 \
                       + flops_conv3_1 + flops_relu5 \
                       + flops_conv1_1 + flops_relu6\
                       + flops_gap\
                       # + flops_batchnorm\
                       +flops_linear1\
                       + flops_relu7 + flops_linear2
                      )
    return total_flops


def memory(image_size, batch):
    """
    Calculates peak memory in bytes.
    Strategy: Sum of ALL activations (input+output for each layer) + Parameters.
    """
    S = image_size
    B = batch
    total_activations_elements = 0
    total_params = 0
    
    def add_conv(cin, cout, k, s_in, stride=1):
        nonlocal total_activations_elements, total_params
        s_out = get_kernel_ops_sout(s_in=s_in, k=k, s=stride)
        
        total_activations_elements += B * cin * s_in * s_in
        total_activations_elements += B * cout * s_out * s_out
        
        total_params += cin * cout * k * k
        
        return cout, s_out

    def add_pool(cin, k, p, s, s_in):
        nonlocal total_activations_elements
        s_out = get_kernel_ops_sout(s_in=s_in, k=k, s=s)

        total_activations_elements += B * cin * s_in * s_in
        total_activations_elements += B * cin * s_out * s_out
        return cout, s_out

    def add_linear(cin, cout):
        nonlocal total_activations_elements, total_params

        total_activations_elements += B * cin
        total_activations_elements += B * cout

        total_params += cin * cout
        return cout    

    def add_gap(cin, s_in):
        nonlocal total_activations_elements
        
        total_activations_elements += B * cin * s_in * s_in
        total_activations_elements += B * cin
        return cin

    cout, h = add_conv(C_IN, 32, 7, S, stride=2)
 
    cout, h = add_pool(cout, 3, 1, 2, h)

    cout, h = add_conv(cout, 64, 5, h, stride=1)
  
    cout, h = add_conv(cout, 128, 3, h, stride=2)
     
    cout, h = add_conv(cout, 256, 1, h, stride=1)

    cout, h = add_conv(cout, 256, 3, h, stride=2)
        
    cout, h = add_conv(cout, 512, 1, h, stride=1)
    
    cout = add_gap(cout, h)

    cout = add_linear(cout, 256)

    cout = add_linear(cout, NUM_CLASSES)
    
    # Total bytes = 4 * (activations + params)
    return 4.0 * (total_activations_elements + total_params)



def latency(image_size, batch, theta):
    """
    theta: {"theta_launch": с, "theta_comp": с/FLOP, "theta_mem": с/байт}
    """
    f = flops(image_size, batch)
    m = memory(image_size, batch)
    return theta["theta_launch"] + theta["theta_comp"] * f + theta["theta_mem"] * m


def energy(image_size, batch, theta_energy):
    """
    energy:  theta_launch + theta_power * latency(S, B)
    """
    t = latency(image_size, batch, theta_energy["latency"])
    return theta_energy["theta_launch"] + theta_energy["theta_power"] * t
    