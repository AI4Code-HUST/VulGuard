from .cli import main, __version__
from .models.deepjit.warper import DeepJIT
from .models.simcom.warper import SimCom
from .models.tlel.warper import TLEL
from .models.lr.warper import LogisticRegression
from .models.lapredict.warper import LAPredict

def init_model(model_name, language, device):   
    if  model_name == "deepjit":
        return DeepJIT(language=language, device=device)
    elif  model_name == "simcom":
        return SimCom(language=language, device=device)
    elif  model_name == "lapredict":
        return LAPredict(language=language)
    elif  model_name == "tlel":
        return TLEL(language=language)
    elif  model_name == "lr":
        return LogisticRegression(language=language)
    else:
        raise Exception("No such model")