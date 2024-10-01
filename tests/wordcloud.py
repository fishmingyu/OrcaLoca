import wordcloud
from datasets import load_dataset


class WordCloud:
    def __init__(self, width=800, height=400, background_color="white"):
        self.width = width
        self.height = height
        self.background_color = background_color
        self.wc = wordcloud.WordCloud(
            width=self.width, height=self.height, background_color=self.background_color
        )

    def generate(self, text):
        self.wc.generate(text)

    def to_file(self, filename):
        self.wc.to_file(filename)


if __name__ == "__main__":
    # Load the dataset
    ds = load_dataset("jiacheng-ye/nl2bash")

    # get training data
    train_data = ds["train"]
    # features ['nl', 'bash']

    # collect the bash commands
    bash_commands = [d["bash"] for d in train_data]

    # split the commands into words
    # split using whitespace
    bash_words = [cmd.split() for cmd in bash_commands]

    # get the first word of each command
    first_words = [cmd[0] for cmd in bash_words]

    # filter out the text larger than 9 characters
    first_words = [word for word in first_words if len(word) < 9]

    # create a word cloud
    wc = wordcloud.WordCloud(width=800, height=400, background_color="white")
    # generate the word cloud
    wc.generate(" ".join(first_words))

    # save the word cloud
    wc.to_file("bash_commands_wordcloud.png")
