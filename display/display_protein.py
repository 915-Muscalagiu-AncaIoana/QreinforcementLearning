from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def show_protein_pretty(X,Y,Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for index in range(len(X)-1):
        x_values = [X[index], X[index+1]]
        y_values = [Y[index], Y[index + 1]]
        z_values = [Z[index], Z[index + 1]]
        plt.plot(x_values,y_values,z_values)
        index+=1

    ax.scatter(X, Y, Z, c='r', marker='o')

    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    plt.show()
