import numpy as np
from matplotlib import pyplot as plt


def loss_function(C, P, xTy, X, Y, r_lambda):
    predict_error = np.square(P - xTy)
    confidence_error = np.sum(C * predict_error)
    regularization = r_lambda * (np.sum(np.square(X)) + np.sum(np.square(Y)))
    total_loss = confidence_error + regularization
    return np.sum(predict_error), confidence_error, regularization, total_loss


def optimize_user(X, Y, C, P, nu, nf, r_lambda):
    yT = np.transpose(Y)
    for u in range(nu):
        Cu = np.diag(C[u])
        yT_Cu_y = np.matmul(np.matmul(yT, Cu), Y)
        lI = np.dot(r_lambda, np.identity(nf))
        yT_Cu_pu = np.matmul(np.matmul(yT, Cu), P[u])
        X[u] = np.linalg.solve(yT_Cu_y + lI, yT_Cu_pu)


def optimize_item(X, Y, C, P, ni, nf, r_lambda):
    xT = np.transpose(X)
    for i in range(ni):
        Ci = np.diag(C[:, i])
        xT_Ci_x = np.matmul(np.matmul(xT, Ci), X)
        lI = np.dot(r_lambda, np.identity(nf))
        xT_Ci_pi = np.matmul(np.matmul(xT, Ci), P[:, i])
        Y[i] = np.linalg.solve(xT_Ci_x + lI, xT_Ci_pi)


def plot_losses(predict_errors, confidence_errors, regularization_list, total_losses):
    plt.subplots_adjust(wspace=100.0, hspace=20.0)
    fig = plt.figure()
    fig.set_figheight(10)
    fig.set_figwidth(10)

    predict_error_line = fig.add_subplot(2, 2, 1)
    confidence_error_line = fig.add_subplot(2, 2, 2)
    regularization_error_line = fig.add_subplot(2, 2, 3)
    total_loss_line = fig.add_subplot(2, 2, 4)

    predict_error_line.set_title("Predict Error")
    predict_error_line.plot(predict_errors)

    confidence_error_line.set_title("Confidence Error")
    confidence_error_line.plot(confidence_errors)

    regularization_error_line.set_title("Regularization")
    regularization_error_line.plot(regularization_list)

    total_loss_line.set_title("Total Loss")
    total_loss_line.plot(total_losses)
    plt.show()


def train():
    predict_errors = []
    confidence_errors = []
    regularization_list = []
    total_losses = []

    for i in range(15):
        if i != 0:
            optimize_user(X, Y, C, P, nu, nf, r_lambda)
            optimize_item(X, Y, C, P, ni, nf, r_lambda)
        predict = np.matmul(X, np.transpose(Y))
        predict_error, confidence_error, regularization, total_loss = loss_function(C, P, predict, X, Y, r_lambda)

        predict_errors.append(predict_error)
        confidence_errors.append(confidence_error)
        regularization_list.append(regularization)
        total_losses.append(total_loss)

        print('----------------step %d----------------' % i)
        print("predict error: %f" % predict_error)
        print("confidence error: %f" % confidence_error)
        print("regularization: %f" % regularization)
        print("total loss: %f" % total_loss)

    predict = np.matmul(X, np.transpose(Y))
    print('final predict')
    print([predict])

    return  predict_errors,  confidence_errors, regularization_list, total_losses


if __name__ == '__main__':
    r_lambda = 40
    nf = 200
    alpha = 40

    R = np.array([[0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
                  [0, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
                  [0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
                  [0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
                  [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
                  [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
                  [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0]])

    nu = R.shape[0]
    ni = R.shape[1]

    # initialize X and Y with very small values
    X = np.random.rand(nu, nf) * 0.01
    Y = np.random.rand(ni, nf) * 0.01

    P = np.copy(R)
    P[P > 0] = 1
    C = 1 + alpha * R

    predict_errors, confidence_errors, regularization_list, total_losses = train()
    plot_losses(predict_errors,  confidence_errors, regularization_list, total_losses)
