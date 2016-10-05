import matplotlib.pyplot as plt

def plot_graphs(train_df):
    plt.figure(1)
    plt.hist(train_df[0], 10)
    plt.ylabel('Feature : 1')

    plt.figure(2)
    plt.hist(train_df[1], 10)
    plt.ylabel('Feature : 2')

    plt.figure(3)
    plt.hist(train_df[2], 10)
    plt.ylabel('Feature : 3')

    plt.figure(4)
    plt.hist(train_df[3], 10)
    plt.ylabel('Feature : 4')

    plt.figure(5)
    plt.hist(train_df[4], 10)
    plt.ylabel('Feature : 5')

    plt.figure(6)
    plt.hist(train_df[5], 10)
    plt.ylabel('Feature : 6')

    plt.figure(7)
    plt.hist(train_df[6], 10)
    plt.ylabel('Feature : 7')

    plt.figure(8)
    plt.hist(train_df[7], 10)
    plt.ylabel('Feature : 8')

    plt.figure(9)
    plt.hist(train_df[8], 10)
    plt.ylabel('Feature : 9')

    plt.figure(10)
    plt.hist(train_df[9], 10)
    plt.ylabel('Feature : 10')

    plt.figure(11)
    plt.hist(train_df[10], 10)
    plt.ylabel('Feature : 11')

    plt.figure(12)
    plt.hist(train_df[11], 10)
    plt.ylabel('Feature : 12')

    plt.figure(13)
    plt.hist(train_df[12], 10)
    plt.ylabel('Feature : 13')

    plt.show()

if __name__ == "__main__":
    plot_graphs()


