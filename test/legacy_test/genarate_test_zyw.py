import numpy as np

# 1,2,3,4,5,6,7,8,9, 10,11,12
# 1,2,3,4,5,6,8,9,10,12,16,17

lst = [1,2,3,4,5,6,8,9,10,12,16,17]
index = 1
for M in {32}:
    for N in {16}:
        for K in lst:
            f = open('./class_test_v2.txt', 'r')
            fw = open('./class_test_new.txt', 'a')
            fw.write("\nclass TestMatmulInt8_{}(TestMatmulInt8):\n".format(index))
            for line in f.readlines()[1:]:
                if "self.x_shape = " in line:
                    fw.write("        self.x_shape = ({}, {})\n".format(str(K), str(M)))
                elif "self.y_shape = " in line:
                    fw.write("        self.y_shape = ({}, {})\n".format(str(N), str(K)))
                elif ("self.trans_x = ") in line:
                    fw.write("        self.trans_x = {}\n".format(str(True)))
                elif ("self.trans_y = ") in line:
                    fw.write("        self.trans_y = {}\n".format(str(True)))
                else:
                    fw.write(line)
            index += 1
            f.close()
            fw.close()

