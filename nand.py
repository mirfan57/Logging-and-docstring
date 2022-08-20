from utils.help_utils import prepare_data, save_plot
from utils.model import Perceptron
import pandas as pd


def main(data, modelName, plotName, eta, epochs):
    df_NAND = pd.DataFrame(data)
    X, y = prepare_data(df_NAND)

    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)

    _ = model.total_loss()

    model.save(filename=modelName, model_dir="model")
    save_plot(df_NAND, model, filename=plotName)

if __name__ == "__main__":
    NAND = {
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y" : [1,1,1,0]
    }
    ETA = 0.3
    EPOCHS = 10
    main(data=NAND, modelName="nand.model", plotName="nand.png", eta=ETA, epochs=EPOCHS)