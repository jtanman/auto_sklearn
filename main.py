from auto_sk.auto_sk import run
from h2o_gbm.h2o_gbm import train_custom_gbm

def main():
    # run auto-sklearn
    # run()

    # run h2o gbm with custom loss
    train_custom_gbm()


if __name__ == "__main__":
    main()
