"""作用：提供 COCO caption 指标评测入口，封装 BLEU、METEOR、ROUGE_L、CIDEr 等 pycocoevalcap scorer。"""

__author__ = 'tylin'
from .tokenizer.ptbtokenizer import PTBTokenizer
from .bleu.bleu import Bleu
from .meteor.meteor import Meteor
from .rouge.rouge import Rouge
from .cider.cider import Cider
from .spice.spice import Spice


class COCOEvalCap:
    """作用：封装 COCO caption 评测流程，负责组织标注、调用 scorer 并汇总整体和逐样本指标。"""
    def __init__(self, coco, cocoRes):
        """作用：初始化对象状态、保存配置参数，并构建后续方法需要使用的成员变量。"""
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': coco.getImgIds()}

    def evaluate(self):
        """作用：对 caption 预测结果执行分词并计算 BLEU、METEOR、ROUGE_L、CIDEr 等 COCO caption 指标。"""
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")]
            # (Spice(), "SPICE")
        

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        """作用：执行 setEval 方法对应的模块内部逻辑，通常由训练、推理或服务流程间接调用。"""
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        """作用：执行 setImgToEvalImgs 方法对应的模块内部逻辑，通常由训练、推理或服务流程间接调用。"""
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        """作用：执行 setEvalImgs 方法对应的模块内部逻辑，通常由训练、推理或服务流程间接调用。"""
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
