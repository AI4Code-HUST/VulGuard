import java.nio.file.Paths

@main def exec(filepath: String, outputDir: String, workspaceName: String) = {
   importCode.c(filepath, workspaceName)
}