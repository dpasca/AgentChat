#==================================================================
# main.py
#
# Author: Davide Pasca, 2024/01/18
# Description: An agent chat app with fact-checking and web search
#==================================================================
import os
import json
import time
from pyexpat.errors import messages
from dotenv import load_dotenv
from OpenAIWrapper import OpenAIWrapper
from datetime import datetime
import pytz # For timezone conversion
import inspect
from StorageLocal import StorageLocal as Storage
from io import BytesIO
from logger import *
from ConvoJudge import ConvoJudge
from OAIUtils import *
import AssistTools

# Load environment variables from .env file
load_dotenv()

import locale
# Set the locale to the user's default setting
locale.setlocale(locale.LC_ALL, '')

USER_BUCKET_PATH = "user_a_00001"

ENABLE_SLEEP_LOGGING = False

ENABLE_WEBSEARCH = True

COL_BLUE = '\033[94m'
COL_YELLOW = '\033[93m'
COL_GREEN = '\033[92m'
COL_ENDC = '\033[0m'
COL_GRAY = '\033[90m'
COL_DRKGRAY = '\033[1;30m'
COL_ENDC = '\033[0m'

#==================================================================
def makeColoredRole(role):
    coloredRole = ''
    if role == "assistant":
        coloredRole = COL_GREEN
    elif role == "user":
        coloredRole = COL_YELLOW
    else:
        coloredRole = COL_BLUE
    coloredRole += role + ">" + COL_ENDC
    return coloredRole

from SessionDict import SessionDict

session = SessionDict(f'_storage/{USER_BUCKET_PATH}/session.json')
logmsg(f"Session: {session}")

#==================================================================
def printChatMsg(msg):
    # Check if msg is a dict with 'role' and 'content' keys
    if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
        for cont in msg['content']:
            str = makeColoredRole(msg['role'])
            str += f" ({cont['type']})" if cont['type'] != "text" else ""
            str += f" {cont['value']}"
            print(str)
    else:
        print(msg)

def inputChatMsg(prompt):
    return input(prompt)

#==================================================================
# Load configuration from config.json
with open('config.json') as f:
    config = json.load(f)

# Load the instructions
with open(config['assistant_instructions'], 'r') as f:
    assistant_instructions = f.read()

META_TAG = "message_meta"

# Special instructions independent of the basic "role" instructions
MESSAGEMETA_INSTUCT = f"""
The user messages usually begins with metadata in a format like this:
<{META_TAG}>
unix_time: 1620000000
</{META_TAG}>
The user does not write this. It's injected by the chat app for the assistant to use.
Do not make any mention of this metadata. Simply use it organically when needed (e.g.
when asked about the time, use the unix_time value but do not mention it explicitly).
"""

FORMAT_INSTRUCT = r"""
When asked about equations or mathematical formulas you should use LaTeX formatting.
For each piece of mathematical content:
 1. If the content is inline, use `$` as prefix and postfix (e.g. `$\Delta x$`)
 2. If the content is a block, use `$$` as prefix and postfix (e.g. `\n$$\sigma = \frac{1}{2}at^2$$\n` here the `\n` are newlines)
"""

# Initialize OpenAI API
_oa_wrap = OpenAIWrapper(api_key=os.environ.get("OPENAI_API_KEY"))

#===============================================================================
def sleepForAPI():
    if ENABLE_LOGGING and ENABLE_SLEEP_LOGGING:
        caller = inspect.currentframe().f_back.f_code.co_name
        line = inspect.currentframe().f_back.f_lineno
        print(f"[{caller}:{line}] sleeping...")
    time.sleep(0.5)

#==================================================================
def prepareUserMessageMeta():
    return f"<{META_TAG}>\nunix_time: {int(time.time())}\n</{META_TAG}>\n"

def stripUserMessageMeta(msg_with_meta):
    msg = msg_with_meta
    begin_tag = f"<{META_TAG}>"
    end_tag = f"</{META_TAG}>"
    end_tag_len = len(end_tag)

    while True:
        start = msg.find(begin_tag)
        if start == -1:
            break
        end = msg.find(end_tag, start)
        if end == -1:
            break

        # Check if the character following the end tag is a newline
        if msg[end + end_tag_len:end + end_tag_len + 1] == "\n":
            msg = msg[:start] + msg[end + end_tag_len + 1:]
        else:
            msg = msg[:start] + msg[end + end_tag_len:]

    return msg

#==================================================================
_judge = ConvoJudge(
    model=config["support_model_version"],
    temperature=config["support_model_temperature"]
    )

# Callback to get the user info from the session
def local_get_user_info():
    # Populate the session['user_info'] with local user info (shell locale and timezone)
    currentTime = datetime.now()
    if not 'user_info' in session:
        session['user_info'] = {}
    session['user_info']['timezone'] = str(currentTime.astimezone().tzinfo)
    return session['user_info']

#==================================================================
# Create the assistant if it doesn't exist
def createAssistant():
    AssistTools.SetSuperGetUserInfoFn(local_get_user_info)

    tools = []
    tools.append({"type": "code_interpreter"})

    # Setup the tools
    for name, defn in AssistTools.ToolDefinitions.items():
        if (not ENABLE_WEBSEARCH) and name == "perform_web_search":
            continue
        tools.append({ "type": "function", "function": defn })

    if config["enable_retrieval"]:
        tools.append({"type": "retrieval"})

    logmsg(f"Tools: {tools}")

    full_instructions = (assistant_instructions
        + "\n" + MESSAGEMETA_INSTUCT
        + "\n" + FORMAT_INSTRUCT)

    codename = config["assistant_codename"]

    # Create or update the assistant
    assist, was_created = _oa_wrap.CreateOrUpdateAssistant(
        name=codename,
        instructions=full_instructions,
        tools=tools,
        model=config["model_version"])

    if was_created:
        logmsg(f"Created new assistant with name {codename}")
    else:
        logmsg(f"Updated existing assistant with name {codename}")

    return assist

# Create the thread if it doesn't exist
def createThread(force_new=False):
    # if there are no messages in the session, add the role message
    if ('thread_id' not in session) or (session['thread_id'] is None) or force_new:
        thread = _oa_wrap.CreateThread()
        logmsg("Creating new thread with ID " + thread.id)
        # Save the thread ID to the session
        session['thread_id'] = thread.id
        session.modified = True
    else:
        thread = _oa_wrap.RetrieveThread(session['thread_id'])
        logmsg("Retrieved existing thread with ID " + thread.id)
    return thread.id

# Local messages management (a cache of the thread)
def getLocMessages():
    # Create or get the session messages list
    return session.setdefault('loc_messages', [])

def appendLocMessage(message):
    getLocMessages().append(message)
    session.modified = True


def messageToLocMessage(message, make_file_url):
    result = {
        "role": message.role,
        "content": []
    }
    for content in message.content:
        if content.type == "text":

            # Strip the message meta if it's a user message
            out_msg = content.text.value
            if message.role == "user":
                out_msg = stripUserMessageMeta(out_msg)

            # Apply whatever annotations may be there
            if content.text.annotations is not None:

                logmsg(f"Annotations: {content.text.annotations}")

                out_msg = ResolveImageAnnotations(
                    out_msg=out_msg,
                    annotations=content.text.annotations,
                    make_file_url=make_file_url)

                out_msg = ResolveCiteAnnotations(
                    out_msg=out_msg,
                    annotations=content.text.annotations,
                    wrap=_oa_wrap)

                out_msg = StripEmptyAnnotationsBug(out_msg)

            result["content"].append({
                "value": out_msg,
                "type": content.type
            })
        elif content.type == "image_file":
            # Append the content with the image URL
            result["content"].append({
                "value": make_file_url(content.image_file.file_id, "image.png"),
                "type": content.type
            })
        else:
            result["content"].append({
                "value": "<Unknown content type>",
                "type": "text"
            })
    return result

#==================================================================
logmsg("Creating storage...")
_storage = Storage("_storage")

# Create the assistant
logmsg("Creating assistant...")
_assistant = createAssistant()

#==================================================================
def make_file_url(file_id, simple_name):
    strippable_prefix = "file-"
    new_name = file_id
    # Strip the initial prefix (if any)
    if new_name.startswith(strippable_prefix):
        new_name = new_name[len(strippable_prefix):]
    new_name += f"_{simple_name}"

    # Out path in the storage is a mix of user ID, file ID and human-readable name
    file_path = f"{USER_BUCKET_PATH}/{new_name}"

    if not _storage.FileExists(file_path):
        logmsg(f"Downloading file {file_id} from source...")
        data = _oa_wrap.GetFileContent(file_id)
        data_io = BytesIO(data.read())
        logmsg(f"Uploading file {file_path} to storage...")
        _storage.UploadFile(data_io, file_path)

    logmsg(f"Getting file url for {file_id}, path: {file_path}")
    return _storage.GetFileURL(file_path)

#==================================================================
def printSummaryAndCritique():
    printChatMsg(f"\n{COL_DRKGRAY}** Summary:")
    printChatMsg(_judge.GenSummary(_oa_wrap))
    printChatMsg(f"\n{COL_DRKGRAY}** Critique:")
    printChatMsg(_judge.GenCritique(_oa_wrap))
    printChatMsg(COL_ENDC)

#==================================================================
def index(do_clear=False):
    # Load or create the thread
    thread_id = createThread(force_new=do_clear)

    _judge.ClearMessages()

    # Always clear the local messages, because we will repopulate
    #  from the thread history below
    session['loc_messages'] = []

    # Get all the messages from the thread
    history = _oa_wrap.ListThreadMessages(thread_id=thread_id, order="asc")
    logmsg(f"Found {len(history.data)} messages in the thread history")
    for (i, msg) in enumerate(history.data):

        # Append message to messages list
        logmsg(f"Message {i} ({msg.role}): {msg.content}")
        appendLocMessage(
            messageToLocMessage(
                message=msg,
                make_file_url=make_file_url))

    printChatMsg(f"Welcome to {config['app_title']}, v{config['app_version']}")
    printChatMsg(f"Assistant: {config['assistant_name']}")

    if (history := getLocMessages()):
        printChatMsg("History:")
        for msg in getLocMessages():
            _judge.AddMessage(msg)
            printChatMsg(msg)

        #printFactCheck(json.loads(_judge.GenFactCheck(_oa_wrap))) # For debug

#==================================================================
def submit_message(assistant_id, thread_id, msg_text):
    msg = _oa_wrap.CreateMessage(
        thread_id=thread_id, role="user", content=msg_text
    )
    run = _oa_wrap.CreateRun(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )
    return msg, run

# Possible run statuses:
#  in_progress, requires_action, cancelling, cancelled, failed, completed, or expired

#==================================================================
def get_thread_status(thread_id):
    data = _oa_wrap.ListRuns(thread_id=thread_id, limit=1).data
    if data is None or len(data) == 0:
        return None, None
    return data[0].status, data[0].id

#==================================================================
def cancel_thread(run_id, thread_id):
    while True:
        run = _oa_wrap.RetrieveRun(run_id=run_id, thread_id=thread_id)
        logmsg(f"Run status: {run.status}")

        if run.status in ["completed", "cancelled", "failed", "expired"]:
            break

        if run.status in ["queued", "in_progress", "requires_action"]:
            logmsg("Cancelling thread...")
            run = _oa_wrap.CancelRun(run_id=run_id, thread_id=thread_id)
            sleepForAPI()
            continue

        if run.status == "cancelling":
            sleepForAPI()
            continue

#==================================================================
def wait_to_use_thread(thread_id):
    for i in range(5):
        status, run_id = get_thread_status(thread_id)
        if status is None:
            return True
        logmsg(f"Thread status from last run: {status}")

        # If it's expired, then we just can't use it anymore
        if status == "expired":
            logerr("Thread expired, cannot use it anymore")
            return False

        # Acceptable statuses to continue
        if status in ["completed", "failed", "cancelled"]:
            logmsg("Thread is available")
            return True

        # Waitable states
        if status in ["queued", "in_progress", "cancelling"]:
            logmsg("Waiting for thread to become available...")

        logmsg("Status in required action: " + str(status == "requires_action"))

        # States that we cannot handle at this point
        if status in ["requires_action"]:
            logerr("Thread requires action, but we don't know what to do. Cancelling...")
            cancel_thread(run_id=run_id, thread_id=thread_id)
            continue

        sleepForAPI()

    return False

#==================================================================
# Handle the required action (function calling)
def handle_required_action(run, thread_id):
    if run.required_action is None:
        logerr("run.required_action is None")
        return

    # Resolve the required actions and collect the results in tool_outputs
    tool_outputs = []
    for tool_call in run.required_action.submit_tool_outputs.tool_calls:
        name = tool_call.function.name

        try:
            arguments = json.loads(tool_call.function.arguments)
        except:
            logerr(f"Failed to parse arguments. function: {name}, arguments: {tool_call.function.arguments}")
            continue

        logmsg(f"Function Name: {name}")
        logmsg(f"Arguments: {arguments}")

        # Look up the function in the dictionary and call it
        if name in AssistTools.ToolActions:
            responses = AssistTools.ToolActions[name](arguments)
        else:
            responses = AssistTools.fallback_tool_function(name, arguments)

        if responses is not None:
            tool_outputs.append(
                {
                    "tool_call_id": tool_call.id,
                    "output": json.dumps(responses),
                }
            )

    # Submit the tool outputs
    logmsg(f"Tool outputs: {tool_outputs}")
    run = _oa_wrap.SubmitToolsOutputs(
        thread_id=thread_id,
        run_id=run.id,
        tool_outputs=tool_outputs,
    )
    logmsg(f"Run status: {run.status}")

#==================================================================
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        end = datetime.now()
        logmsg(f"Function {func.__name__} took {end - start} to complete")
        return result
    return wrapper

#==================================================================
@timing_decorator
def send_message(msg_text):

    thread_id = session['thread_id']

    # Wait or fail if the thread is stuck
    if wait_to_use_thread(thread_id) == False:
        return json.dumps({'replies': []}), 500

    msg_with_meta = prepareUserMessageMeta() + msg_text

    # Add the new message to the thread
    logmsg(f"Sending message: {msg_with_meta}")
    msg, run = submit_message(_assistant.id, thread_id, msg_with_meta)

    # Wait for the run to complete
    last_printed_status = None
    while True:
        run = _oa_wrap.RetrieveRun(thread_id=thread_id, run_id=run.id)
        if run.status != last_printed_status:
            logmsg(f"Run status: {run.status}")
            last_printed_status = run.status

        if run.status == "queued" or run.status == "in_progress":
            sleepForAPI()
            continue

        # Handle possible request for action (function calling)
        if run.status == "requires_action":
            handle_required_action(run, thread_id)

        # See if any error occurred so far
        if run.status is ["expired", "cancelling", "cancelled", "failed"]:
            logerr("Run failed")
            return json.dumps({'replies': []}), 500

        # All good
        if run.status == "completed":
            break

    # Retrieve all the messages added after our last user message
    new_messages = _oa_wrap.ListThreadMessages(
        thread_id=thread_id,
        order="asc",
        after=msg.id
    )
    logmsg(f"Received {len(new_messages.data)} new messages")

    replies = []
    for msg in new_messages.data:
        # Append message to messages list
        locMessage = messageToLocMessage(msg, make_file_url)
        appendLocMessage(locMessage)
        # We only want the content of the message
        replies.append(locMessage)

    logmsg(f"Replies: {replies}")

    if len(replies) > 0:
        logmsg(f"Sending {len(replies)} replies")
        return json.dumps({'replies': replies}), 200
    else:
        logmsg("Sending no replies")
        return json.dumps({'replies': []}), 200

#==================================================================
def printFactCheck(fcReplies):
    #print(fcReplies)
    #return
    if len(fcReplies['fact_check']) == 0:
        return

    for reply in fcReplies['fact_check']:
        #role = reply['role']
        rebuttal = reply.get('rebuttal') or ''
        links = reply.get('links') or []
        if rebuttal == '' and len(links) == 0:
            continue

        outStr = f"\n{COL_DRKGRAY} NOTICE: {rebuttal}"
        if len(links) > 0:
            outStr += "\n"
            for link in links:
                outStr += f"- <{link}>\n"

        printChatMsg(outStr)

#==================================================================
# Main loop for console app
import argparse

def main():

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--clear', action='store_true',
                        help='clear the chat at the start')

    args = parser.parse_args()

    print(f"Logging is {'Enabled' if ENABLE_LOGGING else 'Disabled'}")

    index(args.clear)

    while True:
        # Get user input
        user_input = inputChatMsg(makeColoredRole("user") + " ")

        if user_input == "/clear":
            # Clear the local messages and invalidate the thread ID
            session['loc_messages'] = []
            # Force-create a new thread
            createThread(force_new=True)
            _judge.ClearMessages()
            continue

        # Exit condition (you can define your own)
        if user_input.lower() == '/exit':
            break

        # Send the message to the assistant and get the replies
        replies = json.loads(send_message(user_input)[0])

        # Simulate a response (replace with actual response handling)
        for reply in replies['replies']:
            _judge.AddMessage(reply)
            printChatMsg(reply)

        printFactCheck(json.loads(_judge.GenFactCheck(_oa_wrap)))

if __name__ == "__main__":
    main()
