from utils.options import train_options
from models.baseline_models import Bert, BertCRF, BertBiLstmCRF
from models.siamese_model import CoBertCRF
from utils.dataset import CoNLLDataset
from trainer import TrainerConfig, Trainer
import wandb


def train_config_to_dict(train_config: TrainerConfig):
    return dict((name, getattr(train_config, name)) for name in dir(train_config) if not name.startswith('__'))


if __name__ == "__main__":
    arguments = train_options()
    train_dataset = CoNLLDataset(
        file_path=arguments.file_path,
        viterbi_algorithm=arguments.viterbi,
        encoder_model=arguments.encoder_model
    )
    val_dataset = CoNLLDataset(
        file_path=arguments.file_path.replace('-train.', '-dev.'),
        viterbi_algorithm=arguments.viterbi,
        encoder_model=arguments.encoder_model
    )

    baseline_model = CoBertCRF(
        encoder_model=arguments.encoder_model,
        pretrained_encoder_model_path=arguments.pretrained_path,
        label_to_id=train_dataset.label_to_id
    )

    config = TrainerConfig(viterbi_algorithm=arguments.viterbi, lstm=arguments.lstm)
    wandb.init(project="MultiCoNER", config=train_config_to_dict(config))
    trainer = Trainer(model=baseline_model, config=config, train_dataset=train_dataset, val_dataset=val_dataset)
    trainer.train()
