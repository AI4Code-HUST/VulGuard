import pandas as pd
from .generate_jit_vul_triggering_commit_data import extract_info
from .parser_cpg_data import cpg_data_process
from .generate_ctg_graph import main, ctg_to_csv
from .Main_Trim_CTG import trim

def pipeline(repo_dir, commit_ids, output):
    commit_ids = [pd.read_json(commit_ids, orient="records", lines=True)["commit_id"]]
    jit_vul_triggering_data = extract_info(repo_dir=repo_dir, commit_ids=commit_ids)
    cpg_data_process(jit_vul_triggering_data)
    main()
    ctg_to_csv(f"{output}/ctg.csv")
    trim(f"{output}/ctg.csv", output)
    

    