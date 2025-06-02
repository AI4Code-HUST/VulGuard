import java.nio.file.Paths

@main def exec(filepath: String, outputDir: String, workspaceName: String) = {
   open(workspaceName)
   run.ossdataflow
   val fileName = filepath.split("/").last.toString()
   cpg.graph.E.map(node=>List(node.inNode.id, node.outNode.id, node.label, node.propertiesMap.get("VARIABLE"))).toJson |> outputDir + "/" + fileName + ".edges.json"
   cpg.graph.V.map(node=>node).toJson |> outputDir + "/" + fileName + ".nodes.json"
}