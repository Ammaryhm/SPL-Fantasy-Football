css = """
<style>
    input:focus, textarea:focus {
        border: 1px solid #4CAF50 !important;
        box-shadow: none !important;
        outline: none !important;
    }
    input, textarea {
        border-radius: 6px !important;
        border: 1px solid #444 !important;
        background-color: #1e1e1e !important;
        color: #eee !important;
    }
    .chat-message {
        display: flex;
        align-items: flex-start;
        margin-bottom: 1.5rem;
    }
    .chat-message .avatar {
        flex: 0 0 64px;
        margin-right: 1rem;
    }
    .chat-message .avatar img {
        width: 64px;
        height: 64px;
        border-radius: 50%;
        object-fit: cover;
    }
    .chat-message .message {
        flex: 1;
        background-color: #2a2a2a;
        padding: 1rem;
        border-radius: 10px;
        color: #fff;
        white-space: pre-line;
    }
    .chat-message.user .message {
        background-color: #A100FF;
    }
    .chat-message.bot .message {
        background-color: #2a2a2a;
    }
</style>
"""

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://media.licdn.com/dms/image/v2/D5612AQHnTGzePMwrsw/article-cover_image-shrink_600_2000/article-cover_image-shrink_600_2000/0/1692109901958?e=2147483647&v=beta&t=4hgKSx5kDmscub4rsoIwxbMz0u4l7Z1xWSRG0dgY7MU" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{MSG}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://media.api-sports.io/football/teams/2939.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{MSG}</div>
</div>
'''


