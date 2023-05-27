import os
from dotenv import load_dotenv

load_dotenv()


def main():
  print("Hello World!")
  my_variable = os.getenv('PINECONE_API_KEY')
  print(my_variable)


if __name__ == "__main__":
  main()
