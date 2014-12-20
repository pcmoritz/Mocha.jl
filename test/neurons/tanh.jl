include("../net/gradient-checking.jl")

function test_tanh_neuron(backend::Backend, T, eps)
  println("-- Testing Tanh neuron on $(typeof(backend)){$T}...")

  data = rand(T, 3,4,5,6) - 0.5
  data_blob = make_blob(backend, data)
  neuron = Neurons.Tanh()

  println("    > Forward")
  forward(backend, neuron, data_blob)
  expected_data = tanh(data)
  got_data = zeros(T, size(data))
  copy!(got_data, data_blob)

  @test all(-eps .< got_data - expected_data .< eps)

  println("    > Backward")
  batch_size = 1
  X = randn(10, batch_size)
  Y = ones(1, batch_size)
  data = MemoryDataLayer(name="data", data=Array[X], tops=[:data], batch_size=batch_size)
  label = MemoryDataLayer(name="label", data=Array[Y], tops=[:label], batch_size=batch_size)
  fc1_layer = InnerProductLayer(name="ip1", output_dim=2, neuron=Neurons.Tanh(), bottoms=[:data], tops=[:ip1])
  loss_layer = SoftmaxLossLayer(name="loss", bottoms=[:ip1, :label])
  net = Net("simple", backend, [data, label, fc1_layer, loss_layer])

  copy!(net.states[3].parameters[1].blob, randn(10*2))
  copy!(net.states[3].parameters[2].blob, randn(2))

  test_gradients(net, eps)
end

function test_tanh_neuron(backend::Backend)
  test_tanh_neuron(backend, Float32, 1e-3)
  test_tanh_neuron(backend, Float64, 1e-6)
end

if test_cpu
  test_tanh_neuron(backend_cpu)
end
if test_gpu
  test_tanh_neuron(backend_gpu)
end