def run_conversation(conversation_chain, user_input):
    response = conversation_chain({'question': user_input})
    return response['chat_history']