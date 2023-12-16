# SpotiShuffle

SpotiShuffle is a program that allows you to create new playlists in Spotify based on your existing playlists or starred songs. With SpotiShuffle, you can choose to create playlists with songs ordered randomly or sorted by their similarity.

## How it Works

- When using the "similarity" option, SpotiShuffle will randomly select a starting point song and add it to the new playlist.
- The program will then search for the song most similar to the one just added and add it next.
- This process continues, with SpotiShuffle picking the song most similar to the previously added song until the playlist is complete.
- This way, the new playlist will have songs that flow well together based on their similarity.

## Prerequisites

Before using SpotiShuffle, you'll need to create an app in the Spotify development dashboard. This will provide you with the required credentials to authenticate with the Spotify API. Follow the instructions in the [Spotify Developer Dashboard](https://developer.spotify.com/documentation/web-api/tutorials/getting-started) or watch this [video tutorial](https://www.youtube.com/watch?v=3RGm4jALukM) to learn how to create the app and retrieve the necessary data.

## Setup

1. Clone the repository and navigate to the project directory:

```shell
$ git clone https://github.com/davidefornelli/SpotiShuffle.git
$ cd SpotiShuffle
```

2. Rename the `.example_env` file to `.env` and fill in the credentials retrieved from the Spotify development dashboard:

```shell
$ cp .example_env .env
$ nano .env
```

3. Install the required dependencies using pip:

```shell
$ pip install -r requirements.txt
```

## Usage

To use SpotiShuffle, run the following command:

```shell
$ python spotishuffle.py [options]
```

Options:
- `-p, --playlist_name <name>`: Specify playlist name you want to shuffle. It will use a fuzzy search, and select the first match. If not provided, SpotiShuffle will create a new playlist based on your starred songs.
- `-o, --order <type>`: Specify the order of the songs in the new playlist. Choose between "random" or "similarity" (default choice). 

Example usage:

```shell
$ python spotishuffle.py -p "Led Zeppelin" -o similarity
```

## Feedback and Contributions

If you encounter any issues or have suggestions for improvements, please feel free to [open an issue](https://github.com/your-username/SpotiShuffle/issues). Contributions in the form of pull requests are also welcome!

## License

This project is licensed under the [MIT License](LICENSE).
