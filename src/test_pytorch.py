import torch
from model import LSTM_word2vec
from sklearn.metrics import accuracy_score, classification_report
from train_pytorch import text_field
from train_pytorch import LSTM, test_iter


def load_(load_path, model, opt):
    state_dict = torch.load(load_path, map_location=torch.device('cuda'))
    model.load_state_dict(state_dict["model_state_dict"])
    opt.load_state_dict(state_dict["optimizer_state_dict"])
    return state_dict["valid_loss"]


def evaluate(
    model, test_loader, version="title", threshold=0.5, device=torch.device("cuda")
):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for (labels, (text, text_len)), _ in test_loader:
            labels = labels.to(device)
            text = text.to(device)
            text_len = text_len.to(device)
            output = model(text, text_len)

            output = (output > threshold).int()
            y_pred.extend(output.tolist())
            y_true.extend(labels.tolist())

    print("Classification Report:")
    print(classification_report(y_true, y_pred, labels=[1, 0], digits=4))


if __name__ == "__main__":
    device = torch.device("cuda")
    # model = LSTM_1(text_field=text_field).to(device)
    # model = LSTM().to(device)
    model = LSTM_word2vec(text_field=text_field).to(device)
    # opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    load_("./models/model.pt", model, opt)  # modifies the internal model, and opt

    evaluate(model, test_iter, device=device)
