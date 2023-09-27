import torch
import evaluate
import utils
from consts import consts
from consts import ARGS
import model_att
import model_emb
import model_base
from pathlib import Path


from preprocess import Preprocessor
from preprocess import BaseAnnotator
from preprocess import WikiAnnotator
from preprocess import CoreAnnotator


c = consts()
A = ARGS()

USE_CACHE = False


class Experiment:
    rootdir = Path("../experiments")
    rootdir.mkdir(exist_ok=True)

    def __init__(self):
        self.data_config = c.DATA_CONFIG
        self.path_model_config = c.PATH_MODEL_CONFIG
        self.config = utils.Json.load(self.path_model_config)
        self.config.update(self.data_config.todict())
        self.dir_data = c.DIR_DATA.replace("../", "").replace("/", "")
        self.config_dir = str(self.path_model_config).replace("..", "").replace("/", "")
        # establish experiment folder
        self.exp_name = f"{self.dir_data}-{c.LM_NAME_SUFFIX}-{self.config_dir}"
        # self.exp_name = "/home/thaikami/codes/UCPhrase-exp/data/devdata/fbdata/output_folder"
        if A.exp_prefix:
            self.exp_name += f".{A.exp_prefix}"
        self.dir_exp = self.rootdir / self.exp_name
        self.dir_exp.mkdir(exist_ok=True)
        utils.Json.dump(self.config, self.dir_exp / "config.json")
        print(f"Experiment outputs will be saved to {self.dir_exp}")

        # preprocessor
        self.train_preprocessor = Preprocessor(
            path_corpus=self.data_config.path_train,
            num_cores=c.NUM_CORES,
            use_cache=USE_CACHE,
        )

        # annotator (supervision)
        self.train_annotator: BaseAnnotator = {
            # "wiki": WikiAnnotator(
            #     use_cache=True,
            #     preprocessor=self.train_preprocessor,
            #     path_standard_phrase=self.data_config.path_phrase,
            # ),
            "core": CoreAnnotator(
                use_cache=USE_CACHE, preprocessor=self.train_preprocessor
            ),
        }[self.config["annotator"]]

        # model
        model_prefix = "." + A.model_prefix if A.model_prefix else ""
        model_dir = self.dir_exp / f"model{model_prefix}"
        if self.config["model"] == "CNN":
            model = model_att.AttmapModel(
                model_dir=model_dir,
                max_num_subwords=c.MAX_SUBWORD_GRAM,
                num_BERT_layers=self.config["num_lm_layers"],
            )
            self.trainer = model_att.AttmapTrainer(model=model)
        elif self.config["model"] == "emb":
            model = model_emb.EmbedModel(
                model_dir=model_dir, finetune=self.config["finetune"]
            )
            self.trainer = model_emb.EmbedTrainer(model=model)

    def train(self, num_epochs=20):
        self.train_preprocessor.tokenize_corpus()
        self.train_annotator.mark_corpus()
        path_sampled_train_data = self.train_annotator.sample_train_data()
        self.trainer.train(
            path_sampled_train_data=path_sampled_train_data, num_epochs=num_epochs
        )

    def select_best_epoch(self):
        paths_ckpt = [
            p for p in self.trainer.output_dir.iterdir() if p.suffix == ".ckpt"
        ]
        best_epoch = None
        best_valid_f1 = 0.0
        for p in paths_ckpt:
            ckpt = torch.load(p, map_location="cpu")
            if ckpt["valid_f1"] > best_valid_f1:
                best_valid_f1 = ckpt["valid_f1"]
                best_epoch = ckpt["epoch"]
        utils.Log.info(f"Best epoch: {best_epoch}. F1: {best_valid_f1}")
        return best_epoch

    def predict(self, epoch, for_tagging=False):
        test_preprocessor = None
        if for_tagging:
            test_preprocessor = Preprocessor(
                path_corpus=self.data_config.path_tagging_docs,
                num_cores=c.NUM_CORES,
                use_cache=USE_CACHE,
            )
        else:
            test_preprocessor = Preprocessor(
                path_corpus=self.data_config.path_test,
                num_cores=c.NUM_CORES,
                use_cache=USE_CACHE,
            )

        test_preprocessor.tokenize_corpus()

        # """ Model Predict """
        dir_prefix = "tagging." if for_tagging else "kpcand."
        dir_predict = self.trainer.output_dir / f"{dir_prefix}predict.epoch-{epoch}"
        path_ckpt = self.trainer.output_dir / f"epoch-{epoch}.ckpt"
        model: model_base.BaseModel = (
            model_base.BaseModel.load_ckpt(path_ckpt).eval().to(c.DEVICE)
        )
        path_predicted_docs = model.predict(
            path_tokenized_id_corpus=test_preprocessor.path_tokenized_id_corpus,
            dir_output=dir_predict,
            batch_size=1024,
            use_cache=USE_CACHE,
        )

        # """ Model Decode and Evaluate"""
        dir_decoded = self.trainer.output_dir / f"{dir_prefix}decoded.epoch-{epoch}"
        dir_decoded.mkdir(exist_ok=True)

        if for_tagging:
            path_decoded_doc2sents = model_base.BaseModel.decode(
                path_predicted_docs=path_predicted_docs,
                output_dir=dir_decoded,
                threshold=self.config["threshold"],
                use_cache=USE_CACHE,
                use_tqdm=True,
            )
            evaluator = evaluate.SentEvaluator()
            paths_gold = self.data_config.paths_tagging_human
            print(f"Evaluate {path_decoded_doc2sents}")
            print(
                evaluator.evaluate(path_decoded_doc2sents, paths_doc2golds=paths_gold)
            )
        else:
            path_decoded_doc2cands = model_base.BaseModel.get_doc2cands(
                path_predicted_docs=path_predicted_docs,
                output_dir=dir_decoded,
                expected_num_cands_per_doc=self.data_config.kp_num_candidates_per_doc,
                use_cache=USE_CACHE,
                use_tqdm=True,
            )
            evaluator = evaluate.Evaluator()
            print(f"Evaluate {path_decoded_doc2cands}")
            print(evaluator.evaluate(path_doc2cands=path_decoded_doc2cands))


if __name__ == "__main__":
    exp = Experiment()
    exp.train()
    best_epoch = exp.select_best_epoch()
    exp.predict(epoch=best_epoch, for_tagging=False)
