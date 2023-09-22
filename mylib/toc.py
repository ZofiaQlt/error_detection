import nbformat as nbf

def toc(file_path, rec_path):
# Load the notebook content

    with open(file_path, 'r') as f:
        nb = nbf.read(f, as_version=nbf.NO_CONVERT)

    # Generate the numbered table of contents
    toc = "\n\n---\n# Sommaire\n\n"
    section_number = 1
    subsection_numbers = [0]  # To keep track of subsection numbers
    for cell in nb.cells:
        if cell.cell_type == 'markdown' and cell.source.startswith('#'):
            header_level = cell.source.count('#')
            header_text = cell.source[header_level:].strip()

            if header_level == 1:
                toc += f" {section_number}. {header_text}\n"
                section_number += 1
                subsection_numbers = [0]  # Reset subsection numbers for a new section
            elif header_level == 2:
                subsection_numbers[-1] += 1  # Increment the current subsection number
                toc += f"    - {section_number-1}.{subsection_numbers[-1]}. {header_text}\n"
                
    # Add the dashes at the end of the table of contents
    toc += "\n---"

    # Insert the table of contents at the beginning of the notebook
    nb.cells.insert(0, nbf.v4.new_markdown_cell(toc))

    # Save the modified notebook
    with open(rec_path, 'w') as f:
        nbf.write(nb, f)
