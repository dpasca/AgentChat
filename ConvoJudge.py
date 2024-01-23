#==================================================================
# ConvoJudge.py
#
# Author: Davide Pasca, 2024/01/23
# Description: A judge for conversations
#==================================================================

from OpenAIWrapper import OpenAIWrapper

class ConvoJudge:
    def __init__(self, model, temperature):
        self.srcMessages = []
        self.model = model
        self.temperature = temperature
        self.instructionsForSummary = """
You will receive a conversation between User and Assistant in the format:
- SUMMARY (optional): [Summary of the conversation so far]
- Message: <index> by <role>:\n<content>
- Message: ...

Output a synthesized summary of the conversation in less than 100 words.
Do not prefix with "Summary:" or anything like that, it's implied. 
Output must be optimized for a LLM, human-readability is not important.

Rules for output:
1. Retain key data (names, dates, numbers, stats) in summaries.
2. If large data blocks, condense to essential information only.
"""

        self.instructionsForCritique = """
You will receive a conversation between User and Assistant in the format:
- SUMMARY (optional): [Summary of the conversation so far]
- Message: <index> by <role>:\n<content>
- Message: ...

Assistant is a mind-reading AI based on an LLM. Its goal is to provide total delegation
of the tasks required towards the user's goal.

Generate a critique where Assistant lacked and could have done better towards the goal
of minimizing the user's effort to reach their goal. Be synthetic, direct and concise.
This critique will be related to Assistant, for it to act upon it and improve.
Output must be optimized for a LLM, human-readability not a factor.
Reply in the following format:
{
    "text": <critique text>,
    "requires_action": <true/false>
}
"""

        self.instructionsForFactCheck = """
You will receive a conversation between User and Assistant in the format:
- SUMMARY (optional): [Summary of the conversation so far]
- Message: <index> by <role>:\n<content>
- Message: ...

Perform a fact-check for the last message in the conversation and
output your findings in a fact-check list with the following format:
{
  "fact_check": [
    {
      "role": <role of the assertion>,
      "msg_index": <message index>,
      "applicable": <true/false>,
      "correctness": <degree of correctness, 0 to 5>
      "rebuttal": <extremely short rebuttal, inclusive of references>,
      "links": <list of links to sources>,
    }
  ]
}
Do not produce "rebuttal" or "links" if "applicable" is false.
Beware of the fact that the assisant may have tools that you may not be
aware of, such as access to the Internet and user's details.
"""

    def AddMessage(self, srcMsg):
        self.srcMessages.append(srcMsg)

    def ClearMessages(self):
        self.srcMessages = []

    def buildConvoString(self, maxMessages):
        convo = ""
        n = len(self.srcMessages)
        staIdx = max(0, n - maxMessages)
        for index in range(staIdx, n):
            srcMsg = self.srcMessages[index]
            #convo += "- " + srcMsg['role'] + ": "
            convo += f"- Message: {index} by {srcMsg['role']}:\n"
            for cont in srcMsg['content']:
                convo += cont['value'] + "\n"
        return convo

    def genCompletion(self, wrap, instructions, maxMessages=1000):
        convo = self.buildConvoString(maxMessages)
        #print(f"Sending Conversation:\n{convo}\n------")
        response = wrap.CreateCompletion(
            model=self.model,
            temperature=self.temperature,
            messages=[
            {"role": "system", "content": instructions},
            {"role": "user",   "content": convo}
        ])
        return response.choices[0].message.content

    def GenSummary(self, wrap):
        return self.genCompletion(wrap, self.instructionsForSummary)
    def GenCritique(self, wrap):
        return self.genCompletion(wrap, self.instructionsForCritique)
    def GenFactCheck(self, wrap):
        return self.genCompletion(wrap, self.instructionsForFactCheck, 3)

