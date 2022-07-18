import numpy as np


def svm_loss_naive(W, X, y, reg):
    '''
    Inputs:
      - W: A numpy array of shape (D, C) containing weights.
      - X: A numpy array of shape (N, D) containing a minibatch of data.
      - y: A numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss as single float
      - gradient with respect to weights W; an array of same shape as W
    '''

    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:  # 也就是只有1(xiWj - xiWyi +1>0)的时候才进行loss和gradient的计算
                loss += margin
                dW[:, y[i]] += -X[i, :]
                dW[:, j] += X[i, :]
                # 上面两行代码这样理解，每张图片输入，就是i循环一次，对应的j循环了num_class次，就是把dW矩阵的每一列都遍历到了，（也就是每张图对应一个dW，i循环完了，就是将num_train个dW矩阵叠加了一遍），这里dW[:,y[i]]+= -X[i,:]在每一次i循环都执行，就是第y[i]列是其他列的和，其他列只循环了一次。
                # if margin > 0: 表示有错误的评分出现，这个时候才会产生损失值，所以这里才会有loss的计算

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW = dW / num_train
    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    dW = dW + reg * W
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = X.dot(W)
    # 根据已知的y，找出本应该得分最高的分类得分，存放入一个矩阵，这个矩阵大小【1*N】
    score_correct = scores[np.arange(num_train), y]
    # 必须这样才能把矩阵变为[N*1],不能用转置，python中转置操作对以为矩阵不起作用（有点坑）
    score_correct = np.reshape(score_correct, (num_train, 1))
    # 矩阵化实现(xiW)j−(xyiW)j+Δ
    margin_matrix = scores - score_correct + 1
    # 因为loss公式中指明j=y[i]的列不参与计算，因为这一列是本应该分类正确的，其他列都是和这一列作比较。
    margin_matrix[np.arange(num_train), y] = 0

    margin_matrix[margin_matrix <= 0] = 0
    # 计算所有loss的和
    loss = np.sum(margin_matrix)

    loss /= num_train
    loss = loss + 0.5 * reg * np.sum(W * W)
    # 这一步是对用梯度公式中 1(xiWj - xiWyi +1>0))  这一项
    margin_matrix[margin_matrix > 0] = 1
    # 将所有非正确分类，且这个非正确分类的评分要大于正确的评分，做个求和
    row_sum = np.sum(margin_matrix, axis=1)
    margin_matrix[np.arange(num_train), y] = -row_sum
    # 上面两句代码对应for循环中的dW[:,y[i]]+= -X[i,:]这一句
    dW = np.dot(X.T, margin_matrix) / num_train + reg * W
    # 然后根据矩阵的对应关系进行计算，就是维度关系
    return loss, dW
