from utils.options import train_options
from models.siamese_model import CoBert
from utils.dataset import SiameseDataset
from trainer import TrainerConfig, ContrastiveTrainer
import wandb


def train_config_to_dict(train_config: TrainerConfig):
    return dict((name, getattr(train_config, name)) for name in dir(train_config) if not name.startswith('__'))


if __name__ == "__main__":
    arguments = train_options()
    train_dataset = SiameseDataset(
        file_path=arguments.file_path,
        train_data=True,
        viterbi_algorithm=arguments.viterbi,
        encoder_model=arguments.encoder_model,
        max_pairs=arguments.train_max_pairs,
        max_instances=arguments.max_instances
    )
    val_dataset = SiameseDataset(
        file_path=arguments.file_path.replace('-train.', '-dev.'),
        train_data=False,
        viterbi_algorithm=arguments.viterbi,
        encoder_model=arguments.encoder_model,
        max_pairs=arguments.val_max_pairs,
        max_instances=arguments.max_instances
    )

    model = CoBert(encoder_model=train_dataset.encoder_model, label_to_id=train_dataset.label_to_id)
    config = TrainerConfig(viterbi_algorithm=arguments.viterbi, lstm=arguments.lstm, lr=1e-5, clip_gradients=True)
    wandb.init(project="MultiCoNER", config=train_config_to_dict(config))
    trainer = ContrastiveTrainer(model=model, config=config, train_dataset=train_dataset, val_dataset=val_dataset)
    trainer.train()
