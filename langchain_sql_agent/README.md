# SQL Agent with LangChain

This tutorial illustrates the use of a SQL Agent using LangChain.

To run the jupyter notebook **notebooks\sql_agent_class_example.sql** you have to create a .env file with the environment variables that the LLM you are using requires. 

For example, if you are connecting to OpenAI your env should look like this:

`
OPENAI_API_KEY="YOUR_API_KEY"
`

After you create the env make sure you update its path by setting the variable `env_path` in the file **source\sql_agent.py**

`env_path = "../.env"`

To connect to other LLM providers you may have to modify the **source\sql_agent.py** file to consider the requirements of your provider. 

For additional information about how to build agents in LangChain, please refer to their [official documentation](https://python.langchain.com/docs/tutorials/sql_qa/).
