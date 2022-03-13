#!/usr/bin/env python3
# python chatbot_for_intents.py

# please ensure to train the AI prior to running the ChatBot


class ChatBot:
    """
    A terminal based interactive chatbot that allows the user
    to input sentences into terminal, and the A.I. will try to
    determine the intent of the user input sentence
    """
    
    def __init__(self, ai_model):
        self.exit_commands = ("quit", "exit", "stop")
        self.model = ai_model

    def chat_with_ai_for_intent(self):
        """Commence terminal chat with chatbot for replies of intent"""
        intro = "I'm an A.I. that's trying to figure out the intent "
        intro += "of your question or comment.\nPlease input something:\n~"
        
        inputitem = input(intro)
        self._have_ai_predict_intent(inputitem)
        return

    def _have_ai_predict_intent(self, reply):
        """ user input is AI analyzed and predicted intent is returned """
        while not self._exit_chatbot_intent_determination(reply):
            reply = input(self.model.predict_matching_sentence_response \
                          (reply, quiet=True)+"\n~ ")
        return

    def _exit_chatbot_intent_determination(self, reply):
        """ if an exit command is input, exit the chatting for intents """
        for exit_command in self.exit_commands:
            if exit_command == reply: 
                print("\nWell that was just swell. Till next time!\n")
                return True
        return False
    
    