
#导入训练集
#!/usr/bin/env python3
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
