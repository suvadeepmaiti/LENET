# A file created for the model to fit arbitary input which validates learning, if it converges to accuracy 1.0
from lenet import Lenet_SMAI
import numpy as np
import time

if __name__ == "__main__":
    np.random.seed(10)
    model = Lenet_SMAI()
    model.summary()
    model.compile_adam()
    batch_size = 32
    itr = 500
    st = time.time()
    img, label = np.random.rand(batch_size, 32, 32, 1), np.random.randint(0, 10,(batch_size,))
    for i in range(itr):
        # a sample way of doing a train step
        loss = model(img, label)
        print(loss)
        grads = model.compute_gradients()
        model.apply_gradients(grads)
    print(f"accuracy : {np.count_nonzero(model(img, mode='test') == label) / batch_size}")
    print(f'took {time.time() - st} for {itr} batch steps of size {batch_size}, 1 prediction')