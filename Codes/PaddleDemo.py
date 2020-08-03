import paddle.fluid as fluid

x1 = fluid.layers.fill_constant(shape=[2, 2], value=1, dtype='int64')
x2 = fluid.layers.fill_constant(shape=[2, 2], value=1, dtype='int64')

y1 = fluid.layers.sum(x=[x1, x2])

place = fluid.CPUPlace()
exe = fluid.executor.Executor(place)
exe.run(fluid.default_startup_program())

result = exe.run(program=fluid.default_main_program(),
                 fetch_list=[y1])
print(result)
