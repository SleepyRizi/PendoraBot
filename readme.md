HOW TO INSTALL/USE (**google collab**):
1) run notebook at [manuals/llm_telegram_bot_cublas.ipynb](https://colab.research.google.com/drive/1XJDR1Hb1K-ObuLl58vTJN_3mQlaMWyFW?usp=sharing)
2) install, set bot token, run
---------------




CONFIGURATION:

`app_config.json` - config for running as standalone app (`run.sh` or `run.cmd`)  
`ext_config.json` - config for running as extension

```
x_config.json
    bot_mode=admin  
        specific bot mode. admin for personal use
            - admin - bot answer for everyone in chat-like mode. All buttons, include settings-for-all are avariable for everyone. (Default)
            - chat - bot answer for everyone in chat-like mode. All buttons, exclude settings-for-all are avariable for everyone. (Recommended for chatting)
            - chat-restricted - same as chat, but user can't change default character
            - persona - same as chat-restricted, but reset/regenerate/delete message are unavailable too. 
            - notebook - notebook-like mode. Prefixes wont added automaticaly, only "\n" separate user and bot messages. Restriction like chat mode.
            - query - same as notebook, but without history. Each question for bot is like new convrsation withot influence of previous questions
    user_name_template=
        user name template, useful for group chat.
        if empty bot always get default name of user - You. By default even in group chats bot perceive all users as single entity "You"
        but it is possible force bot to perceive telegram users names with templates: 
            FIRSTNAME - user first name (Jon)
            LASTNAME - user last name (Dow)
            USERNAME - user nickname (superguy)
            ID - user Id (999999999)
        so, user_name_template="USERNAME FIRSTNAME ID" translatede to user name "superguy Jon 999999999"
        but if you planed to use template and group chat - you shold add "\n" sign to stopping_strings to prevent bot impersonating!!!
    generator_script=GeneratorLlamaCpp
        name of generator script (generators folder):
            - generator_exllama - based on llama-cpp-python, recommended
            - generator_llama_cpp - based on llama-cpp-python, recommended
            - generator_langchain_llama_cpp - based in langchain+llama
            - generator_transformers - based on transformers, untested
            - generator_text_generator_webui_openapi - use oobabooga/text-generation-webui OpenAPI extension
            - (BROKEN) generator_text_generator_webui - module to integrate in oobabooga/text-generation-webui (curently broken:( )
            - (OUTDATED) generator_text_generator_webui_api - use oobabooga/text-generation-webui API (old api version)
    model_path=models\llama-13b.ggml.q4_0.gguf
        path to model file or directory
    characters_dir_path=characters
    default_char=Kaki-2.json
        default cahracter and path to cahracters folder
    presets_dir_path=presets
    default_preset=Shortwave.yaml
        default generation preset and path to preset folder
    model_lang=en
    user_lang=en
        default model and user language. User language can be switched by users, individualy.
    html_tag_open=<pre>
    html_tag_close=</pre>
        tags for bot answers in tg. By default - preformatted text (pre)
    history_dir_path=history
        directory for users history
    token_file_path=configs\\telegram_token.txt
        bot token. Ask https://t.me/BotFather
    admins_file_path=configs\\telegram_admins.txt
        users whos id's in admins_file switched to admin mode and can choose settings-for-all (generating settings and model)
    users_file_path=configs\\telegram_users.txt
        if just one id in users_file - bot will ignore all users except this id (id's). Even admin will be ignored
    generator_params_file_path=configs\\telegram_generator_params.json
        default text generation params, overwrites by choosen preset 
    user_rules_file_path=configs\\telegram_user_rules.json
        user rules matrix 
    telegram_sd_config=configs\\telegram_sd_config.json
        stable diffusion api config
    stopping_strings=<END>,<START>,end{code}
        generating settings - which text pattern stopping text generating? Add "\n" if bot sent too much text.
    eos_token=None
        generating settings
    translation_as_hidden_text=on
        if "on" and model/user lang not the same - translation will be writed under spoiler. If "off" - translation without spoiler, no original text in message.
    sd_api_url="http://127.0.0.1:7860"
    stable diffusion api url, need to use "photo" prefixes
    proxy_url
        to avoid provider blocking


generator_params.json
    config for generator 

sd_config.json
    config for stable diffusion

telegram_admins.txt
    list of users id who forced to admin mode. If telegram_users not empty - must be in telegram_users too!

telegram_users.txt
    list og users id (or groups id) who permitted interact with bot. If empty - everyone permitted

telegram_token.txt (you can use .env instead)
    telegram bot token

telegram_user_rules.json
    buttons visibility config for various bot modes

```
