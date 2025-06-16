from .deepjit.warper import DeepJIT
from .simcom.warper import SimCom
from .tlel.warper import TLELModel
from .lr.warper import LogisticRegression
from .lapredict.warper import LAPredict
from .jitfine.warper import JITFine
from .vccfinder.warper import VCCFinder

models = ["deepjit", "simcom", "lapredict", "tlel", "lr", "jitfine", "vccfinder"]

def init_model(model_name, language, device):   
    if  model_name == "deepjit":
        return DeepJIT(language=language, device=device)
    elif  model_name == "simcom":
        return SimCom(language=language, device=device)
    elif  model_name == "lapredict":
        return LAPredict(language=language)
    elif  model_name == "tlel":
        return TLELModel(language=language)
    elif  model_name == "lr":
        return LogisticRegression(language=language)
    elif model_name == "jitfine":
        return JITFine(language=language, device=device)
    elif model_name == "vccfinder":
        return VCCFinder(language=language)
    else:
        raise Exception("No such model")