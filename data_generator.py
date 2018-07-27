from skimage import io
import numpy as np
#生成器generator
def generate_for_kp(file_list, label_list, batch_size):
    while True:
        count = 0
        x, y = [], []
        for i,path in enumerate(file_list):
            img=io.imread(path)
            img = np.array(img)
            x_temp=img/255.0
            y_temp=(label_list[i,:]-48.0)/48.0
            count += 1
            x.append(x_temp)
            y.append(y_temp)
            if count % batch_size == 0 and count != 0:
                x = np.array(x)
                x = x.reshape(batch_size, 96, 96, 1).astype("float32")
                y = np.array(y)
                yield x, y
                x, y = [], []