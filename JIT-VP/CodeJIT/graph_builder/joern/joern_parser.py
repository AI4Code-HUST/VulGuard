from config import BASE_DIR
from file_manager import get_outer_dir, join_path, is_path_exist, mkdir_if_not_exist, get_file_name, remove_dir, unlink, write_file
from helpers import get_logger, subprocess_cmd, get_current_timestamp, encode_special_characters_with_html_rules
import os
CURRENT_DIR = get_outer_dir(__file__)
JOERN_IMPORT_PATH = join_path(CURRENT_DIR, "joern_scripts", "import_code.scala")
JOERN_SCRIPT_PATH = join_path(CURRENT_DIR, "joern_scripts", "get_func_graph.scala")
JOERN_SCRIPT_FUNCTION_PATH = join_path(CURRENT_DIR, "joern_scripts", "get_func_graph_parse_function.scala")
logger = get_logger(__name__)



def run_joern(file_path: str, output_dir: str):
    """Extract graph using most recent Joern."""
    if is_path_exist(join_path(output_dir, get_file_name(file_path) + ".nodes.json")):
        return

    logger.info(f"Exporting joern graph from [{file_path}]")
    mkdir_if_not_exist(output_dir)
    
    workspace_name = str(get_current_timestamp()) + "__" + "__".join(file_path.split('/'))
    params = f"filepath={file_path},outputDir={output_dir},workspaceName={workspace_name}"
    command = f"joern --script {JOERN_IMPORT_PATH} --params='{params}'"
    stdout, stderr = subprocess_cmd(command)

    command = f"joern --script {JOERN_SCRIPT_PATH} --params='{params}'"
    print(command)
    logger.debug(command)
    stdout, stderr = subprocess_cmd(command)
    if "script finished successfully" not in stderr:
        logger.warning(f"[{file_path}]{stderr}")

    workspace_dir = join_path(BASE_DIR, "workspace", encode_special_characters_with_html_rules(workspace_name))
    logger.info(f"Removing {workspace_dir}")
    try:
        remove_dir(workspace_dir)
    except Exception as e:
        logger.warning(f"Failed to remove workspace {workspace_dir}")


def run_joern_text(function_text, output_dir, fileName = ""):
    if is_path_exist(join_path(output_dir, fileName + ".cpp.nodes.json")):
        print("Already parsed")
        return
    if function_text is None or isinstance(function_text,float):
        node_p = join_path(output_dir, fileName + ".cpp.nodes.json")
        edge_p = join_path(output_dir, fileName + ".cpp.edges.json")
        write_file(node_p,"[]")
        write_file(edge_p,"[]")
        return
    mkdir_if_not_exist(output_dir)
    cm_id = output_dir.rsplit("/")[-1]
    workspace_name = cm_id + str(get_current_timestamp())+fileName
    tmp_file = join_path(output_dir,fileName+".cpp")
    if not is_path_exist(tmp_file):
        with open(tmp_file, "w+") as f:
            f.write(function_text)
    mkdir_if_not_exist(output_dir)
    logger.info(f"Exporting joern graph [{output_dir}]")
    params = f"filepath={tmp_file},outputDir={output_dir},workspaceName={workspace_name}"
    
    command = f"joern --script {JOERN_IMPORT_PATH} --params='{params}'"
    stdout, stderr = subprocess_cmd(command)
    
    command = f"joern --script {JOERN_SCRIPT_FUNCTION_PATH} --params='{params}'"
    logger.debug(command)
    stdout, stderr = subprocess_cmd(command)
    if "script finished successfully" not in stderr:
        logger.warning(f"[{output_dir}]{stderr}")
    workspace_dir = join_path(BASE_DIR, "workspace", workspace_name)
    try:
        remove_dir(workspace_dir)
    except Exception as e:
        logger.warning(f"Failed to remove workspace {workspace_dir}")
    