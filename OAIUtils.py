#==================================================================
# OAIUtils.py
#
# Author: Davide Pasca, 2024/01/23
# Description: Utilities to manage OpenAI API
#==================================================================

from logger import *
import re

def IsImageAnnotation(a):
    return a.type == "file_path" and a.text.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp"))

# Replace the file paths with actual URLs
def ResolveImageAnnotations(out_msg, annotations, make_file_url):
    new_msg = out_msg
    # Sort annotations by start_index in descending order
    sorted_annotations = sorted(annotations, key=lambda x: x.start_index, reverse=True)

    for a in sorted_annotations:
        if IsImageAnnotation(a):
            file_id = a.file_path.file_id

            logmsg(f"Found file {file_id} associated with '{a.text}'")

            # Extract a "simple name" from the annotation text
            # It's likely to be a full-pathname, so we just take the last part
            # If there are no slashes, we take the whole name
            simple_name = a.text.split('/')[-1] if '/' in a.text else a.text
            # Replace any characters that are not alphanumeric, underscore, or hyphen with an underscore
            simple_name = re.sub(r'[^\w\-.]', '_', simple_name)

            file_url = make_file_url(file_id, simple_name)

            logmsg(f"Replacing file path {a.text} with URL {file_url}")

            # Replace the file path with the file URL
            new_msg = new_msg[:a.start_index] + file_url + new_msg[a.end_index:]

    return new_msg

def ResolveCiteAnnotations(out_msg, annotations, wrap):
    citations = []
    for index, a in enumerate(annotations):

        #if IsImageAnnotation(a):
        #    continue

        logmsg(f"Found citation '{a.text}'")
        logmsg(f"out_msg: {out_msg}")
        # Replace the text with a footnote
        out_msg = out_msg.replace(a.text, f' [{index}]')

        logmsg(f"out_msg: {out_msg}")

        # Gather citations based on annotation attributes
        if (file_citation := getattr(a, 'file_citation', None)):
            logmsg(f"file_citation: {file_citation}")
            cited_file = wrap.client.files.retrieve(file_citation.file_id)
            citations.append(f'[{index}] {file_citation.quote} from {cited_file.filename}')
        elif (file_path := getattr(a, 'file_path', None)):
            logmsg(f"file_path: {file_path}")
            cited_file = wrap.client.files.retrieve(file_path.file_id)
            citations.append(f'[{index}] Click <here> to download {cited_file.filename}')
            # Note: File download functionality not implemented above for brevity

    # Add footnotes to the end of the message before displaying to user
    if len(citations) > 0:
        out_msg += '\n' + '\n'.join(citations)

    return out_msg

# Deal with the bug where empty annotations are added to the message
# We go and remove all 【*†*】blocks
def StripEmptyAnnotationsBug(out_msg):
    # This pattern matches 【*†*】blocks
    pattern = r'【\d+†.*?】'
    # Remove all occurrences of the pattern
    return re.sub(pattern, '', out_msg)

