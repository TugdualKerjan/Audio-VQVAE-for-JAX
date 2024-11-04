import pprint as pp
import nbformat
from nbconvert import PythonExporter
import os

class CustomPythonExporter(PythonExporter):
    def from_notebook_node(self, notebook_node, resources=None, **kw):
        # Call the parent method to get the initial script
        script, resources = super().from_notebook_node(notebook_node, resources, **kw)
        
        # Remove input prompts (e.g., # In[1]:)
        lines = script.splitlines()
        filtered_lines = [line for line in lines if not line.startswith('# In[')]
        
        return '\n'.join(filtered_lines), resources

# Load the notebook
for notebook_filename in os.listdir("."):
    if "ipynb" in notebook_filename:

        with open(notebook_filename, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)

        # Filter cells by tags
        notebook.cells = [cell for cell in notebook.cells 
                            if 'tags' in cell.metadata and 'export' in cell.metadata['tags']]

        # Use the custom exporter
        exporter = CustomPythonExporter()
        exporter.exclude_output_stdin = True
        exporter.exclude_output_prompt = True
        script, _ = exporter.from_notebook_node(notebook)

        # Save the exported Python script
        output_filename = notebook_filename[:-6] + ".py"
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(script)

        print(f"Exported cells with tag 'export' to {output_filename}")
