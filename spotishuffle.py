import click
from datetime import datetime
from dotenv import load_dotenv
from thefuzz import process
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
from typing import Dict
from typing import List
from urllib import parse
import logging
import numpy as np
import pandas as pd
import random

logging.basicConfig()
logging.root.setLevel(logging.INFO)

load_dotenv(override=True)

def fetch_starred_songs(
        spotipy_client: Spotify
    ) -> List[Dict]:
    """
    Retrieves all starred (liked) songs for the current authenticated Spotify user.

    This function will continue to fetch liked songs in batches using the Spotify API until all liked songs are retrieved.

    Parameters:
    spotipy_client (Spotify): An authenticated instance of the Spotify client to make API calls.

    Returns:
    list: A list containing all the liked songs of the current user.
    """

    # Log the start of the fetching process for starred songs.
    logging.info("Fetching starred songs")

    # Initialize the batch size and starting offset for API pagination.
    limit = 50
    offset = 0

    # List to store aggregated results from the API responses.
    results = []

    # Loop indefinitely until there are no more liked songs to fetch.
    while True:
        # Fetch a batch of liked songs using the current offset and limit.
        result = spotipy_client.current_user_saved_tracks(offset=offset, limit=limit)

        # Append the current batch of liked songs to the aggregate list.
        results.append(result)

        # Log the URL of the next set of liked songs, if available.
        logging.debug(result.get('next'))

        # Check if there is a 'next' URL to fetch more liked songs; if so, update the offset and limit.
        if result.get('next'):
            nxt_params = parse.parse_qs(parse.urlsplit(result['next']).query)
            limit = int(nxt_params['limit'][0])
            offset = int(nxt_params['offset'][0])
        else:
            # If there is no 'next' URL, break out of the loop as we have fetched all liked songs.
            break

    # List to hold the final set of liked songs.
    songs = []

    # Iterate over the aggregated API responses and extract the song items.
    for r in results:
        songs.extend(r['items'])

    # Return the final list of liked songs.
    return songs


def fetch_playlist_songs(
        spotipy_client: Spotify,
        playlist: Dict
    ) -> List[Dict]:
    """
    Retrieves the songs for the given playlist.

    This function will continue to fetch liked songs in batches using the Spotify API until all songs are retrieved.

    Parameters:
    spotipy_client (Spotify): An authenticated instance of the Spotify client to make API calls.
    playlis (Dict): The playlist object

    Returns:
    list: A list containing all the playlist songs.
    """

    # Log the start of the fetching process for starred songs.
    logging.info(f"Fetching playlist songs: {playlist['name']}")

    # Initialize the batch size and starting offset for API pagination.
    limit = 100
    offset = 0

    # List to store aggregated results from the API responses.
    results = []

    # Loop indefinitely until there are no more songs to fetch.
    while True:
        # Fetch a batch of liked using the current offset and limit.
        result = spotipy_client.playlist_items(
            playlist_id=playlist['id'],
            offset=offset,
            limit=limit
        )

        # Append the current batch of songs to the aggregate list.
        results.append(result)

        # Log the URL of the next set of songs, if available.
        logging.debug(result.get('next'))

        # Check if there is a 'next' URL to fetch more liked songs; if so, update the offset and limit.
        if result.get('next'):
            nxt_params = parse.parse_qs(parse.urlsplit(result['next']).query)
            limit = int(nxt_params['limit'][0])
            offset = int(nxt_params['offset'][0])
        else:
            # If there is no 'next' URL, break out of the loop as we have fetched all liked songs.
            break

    # List to hold the final set of liked songs.
    songs = []

    # Iterate over the aggregated API responses and extract the song items.
    for r in results:
        songs.extend(r['items'])

    # Return the final list of liked songs.
    return songs


def fetch_playlists(
        spotipy_client: Spotify,
        user_id: str
    ) -> List[Dict]:
    """
    Retrieves all playlists for a given Spotify user.

    This function will continue to fetch playlists in batches using the Spotify API until all playlists are retrieved.

    Parameters:
    spotipy_client (Spotify): An authenticated instance of the Spotify client to make API calls.
    user_id (str): The Spotify user ID for which to retrieve playlists.

    Returns:
    list: A list containing all the playlists of the specified user.
    """

    # Log the start of the playlist fetching process.
    logging.info("Fetching playlists")

    # Initialize the batch size and starting offset for API pagination.
    limit = 50
    offset = 0

    # List to store aggregated results from the API responses.
    api_u_playlists = []

    # Loop indefinitely until there are no more playlists to fetch.
    while True:
        # Fetch a batch of playlists using the current offset and limit.
        pls = spotipy_client.user_playlists(
            user=user_id,
            offset=offset,
            limit=limit
        )

        # Append the current batch of playlists to the aggregate list.
        api_u_playlists.append(pls)

        # Log the URL of the next set of playlists, if available.
        logging.debug(pls.get('next'))

        # Check if there is a 'next' URL to fetch more playlists; if so, update the offset and limit.
        if pls.get('next'):
            nxt_params = parse.parse_qs(parse.urlsplit(pls['next']).query)
            limit = int(nxt_params['limit'][0])
            offset = int(nxt_params['offset'][0])
        else:
            # If there is no 'next' URL, break out of the loop as we have fetched all playlists.
            break

    # List to hold the final set of playlists.
    playlists = []

    # Iterate over the aggregated API responses and extract the playlist items.
    for ap in api_u_playlists:
        playlists.extend(ap['items'])

    # Return the final list of playlists.
    return playlists


def spotify_randomize_songs(
        spotify_songs_uris: List[str]
    ) -> List[str]:
    """
    Randomizes the order of Spotify song URIs in the input list.

    Args:
        spotify_songs_uris (List[str]): A list of Spotify song URIs to be randomized.

    Returns:
        List[str]: A new list containing the input song URIs in a randomized order.
    """

    # Log the action of randomizing songs to help with debugging or tracking application behavior.
    logging.info("Randomizing songs")

    # Make a copy of the input list to avoid modifying the original list of song URIs.
    n_songs = spotify_songs_uris.copy()

    # Use the 'shuffle' method provided by the 'random' module to randomly reorder the elements of the copied list.
    random.shuffle(n_songs)

    # Return the shuffled (randomized) list of song URIs.
    return n_songs


def create_playlist(
    spotipy_client: Spotify,
    playlist_name: str,
    spotify_songs_uris: List[str]
) -> None:
    """
    Creates a new Spotify playlist or updates an existing one with the given tracks.

    This function checks if a playlist with the specified name already exists for the current user.
    If it does, it updates the playlist with the provided track URIs. If not, it creates a new
    playlist and populates it with the tracks.

    Args:
        spotipy_client (Spotify): An authenticated instance of the Spotify client.
        playlist_name (str): The name of the playlist to create or update.
        spotify_songs_uris (List[str]): A list of Spotify track URIs to populate the playlist with.

    Returns:
        None
    """

    # Retrieve the current user's information
    c_user = spotipy_client.current_user()
    
    # Fetch the current user's playlists
    playlists = fetch_playlists(
        spotipy_client=spotipy_client,
        user_id=c_user['id']
    )
    
    # Log the attempt to push the playlist
    logging.info(f'Pushing playlist: {playlist_name}')
    
    # Check if the playlist already exists
    playlist_new = [p for p in playlists if p['name'] == playlist_name]
    playlist_new = playlist_new[0] if playlist_new else None

    # Create the playlist if it doesn't exist
    if not playlist_new:
        playlist_new = spotipy_client.user_playlist_create(
            user=c_user['id'],
            name=playlist_name,
            public=False
        )

    # Initialize pagination variables
    limit = 50
    offset = 0
    api_playlists_items = []
    
    # Paginate through the playlist items
    while True:
        pit = spotipy_client.playlist_items(
            playlist_id=playlist_new['id'],
            offset=offset,
            limit=limit
        )
        api_playlists_items.append(pit)
        
        # Log the next page URL for debugging purposes
        logging.debug(pit.get('next'))
        
        # Check if there is a next page and update pagination variables
        if pit.get('next'):
            nxt_params = parse.parse_qs(parse.urlsplit(pit['next']).query)
            limit = int(nxt_params['limit'][0])
            offset = int(nxt_params['offset'][0])
        else:
            break

    # Flatten the paginated results into a single list of items
    playlist_items = []
    for ap in api_playlists_items:
        playlist_items.extend(ap['items'])

    # Clean the playlist by removing all occurrences of current items
    chunks = [playlist_items[x:x+100] for x in range(0, len(playlist_items), 100)]
    for ch in chunks:
        spotipy_client.playlist_remove_all_occurrences_of_items(
            playlist_id=playlist_new['id'],
            items=[t['track']['uri'] for t in ch]
        )

    # Add the new items to the playlist in chunks of 100
    t_chunks = [spotify_songs_uris[x:x+100] for x in range(0, len(spotify_songs_uris), 100)]
    for t_ch in t_chunks:
        spotipy_client.playlist_add_items(
            playlist_id=playlist_new['id'],
            items=t_ch
        )


def similarity_sorter(
    sp: Spotify,
    spotify_songs: List[dict],
    start_index = 0
) -> List[str]:
    """
    Orders a list of Spotify songs by their audio feature similarity.

    This function takes a list of Spotify song dictionaries, extracts their audio features using the Spotify API,
    and then orders them by similarity using cosine distance on the normalized features.

    Args:
        sp (Spotify): An authenticated instance of the Spotify client.
        spotify_songs (List[dict]): A list of dictionaries where each dictionary represents a Spotify song.
        start_index (int, optional): The index of the first song to start ordering from. Defaults to 0.

    Returns:
        List[str]: A list of Spotify song URIs ordered by audio feature similarity.

    Note:
        - The Spotify song dictionaries in `spotify_songs` are expected to have a 'track' key with a nested 'uri' key.
        - The similarity is calculated based on a predefined set of audio features provided by the Spotify API.
        - The function uses MinMaxScaler for normalization and pairwise_distances for calculating cosine distances.
    """
    # Log the process of song similarity calculation
    logging.info("Processing songs similarity")

    # Split the list of songs into chunks of 100 to comply with the Spotify API's rate limits
    chunks = [spotify_songs[x:x+100] for x in range(0, len(spotify_songs), 100)]

    # Retrieve audio features for all songs in chunks
    songs_features = []
    for ch in chunks:
        songs_features.extend(
            sp.audio_features(tracks=ch)
        )

    # Filter out None values from the list of song features
    songs_features = [i for i in songs_features if i is not None]

    # Create a DataFrame from the song features and set the URI as the index
    dt_songs_features = pd.DataFrame(songs_features)
    dt_songs_features.set_index('uri', inplace=True)

    # Select relevant audio features for similarity comparison
    dt_songs_features = dt_songs_features.filter(
        [
            'acousticness',
            'danceability',
            'energy',
            'instrumentalness',
            'key',
            'loudness',
            'liveness',
            'mode',
            'speechiness',
            'tempo',
            'time_signature',
            'valence'
        ]
    )

    # Normalize the features and calculate pairwise cosine distances between songs
    scaler = MinMaxScaler()
    songs_dist = pairwise_distances(scaler.fit_transform(dt_songs_features), metric="cosine")

    # Convert the distances matrix to a DataFrame and replace zeros with NaN
    dt_songs_dist = pd.DataFrame(songs_dist, columns=dt_songs_features.index, index=dt_songs_features.index).replace(0, np.nan)

    # Make a copy of the distances DataFrame for manipulation
    dt_songs_dist_cp = dt_songs_dist.copy()

    # Initialize the list of ordered songs starting with the song at the given start index
    songs_new_order = []
    first_id = spotify_songs[start_index]
    songs_new_order.append(first_id)

    # Order the songs by finding the closest song to the current one until all songs are ordered
    while True:
        id_to_drop = first_id
        second_id = dt_songs_dist_cp[first_id][dt_songs_dist_cp[first_id] == dt_songs_dist_cp[first_id].min()].index[0]
        songs_new_order.append(second_id)
        dt_songs_dist_cp.drop(id_to_drop, axis=1, inplace=True)
        dt_songs_dist_cp.drop(id_to_drop, axis=0, inplace=True)
        first_id = second_id

        # Break the loop when only one song remains
        if dt_songs_dist_cp.shape[0] == 1:
            break
    
    # Return the ordered list of song URIs
    return songs_new_order


def playlist_mixer(
    sp: Spotify,
    tracks_uris: List[str],
    order='similarity'
) -> List[str]:
    """
    Shuffles and optionally orders a list of Spotify track URIs, then creates a playlist.

    This function shuffles the provided list of Spotify track URIs using a custom randomization
    function. Depending on the 'order' parameter, it can also reorder the shuffled tracks by
    similarity. It then creates a new Spotify playlist with these tracks.

    Args:
        sp (Spotify): An authenticated instance of the Spotify client.
        tracks_uris (List[str]): A list of Spotify track URIs to be shuffled.
        order (str, optional): Determines the ordering of the shuffled tracks in the playlist.
                               'random' for random order, 'similarity' for similarity-based order.
                               Defaults to 'similarity'.

    Returns:
        bool: True if the playlist was created and pushed successfully, False otherwise.
    """

    # Log the start of the shuffling process with a current timestamp.
    logging.info("Shuffling started at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Shuffle songs using a custom function that randomizes the order of provided track URIs.
    logging.info("Randomizing song tracks")
    spotify_songs_shuffled = spotify_randomize_songs(spotify_songs_uris=tracks_uris)

    # Decide on the naming and ordering of the playlist based on the 'order' parameter.
    if order == 'random':
        spotify_songs_uris = spotify_songs_shuffled
    elif order == 'similarity':
        # Order the shuffled songs by their similarity.
        logging.info("Ordering songs by similarity")
        spotify_songs_uris = similarity_sorter(
            sp=sp,
            spotify_songs=spotify_songs_shuffled
        )
    else:
        raise ValueError("Unsupported order. Supported orders are 'similarity' and 'random'.")

    return spotify_songs_uris


def spotify_login() -> SpotifyOAuth:
    """
    Creates a Spotify API handler with OAuth authentication.

    This function initializes the SpotifyOAuth object which is used to perform authenticated operations with the Spotify API.

    Returns:
    SpotifyOAuth: An instance of the SpotifyOAuth class for handling OAuth authentication.
    """

    # Log the creation of the Spotify API handler.
    logging.info("Creating spotify api handler")

    # Define the required scope for accessing user's library and private playlists.
    scope = 'user-library-read playlist-modify-private playlist-read-private'

    # Create an instance of SpotifyOAuth with the specified scope.
    at = SpotifyOAuth(
            scope=scope,
        )

    # Return the SpotifyOAuth instance.
    return at


def spotify_api() -> Spotify:
    """
    Initializes and returns a Spotify client instance with authentication.

    This function handles the creation of a Spotify client object by first authenticating through the `spotify_login` function. The authenticated client can then be used to interact with the Spotify Web API.

    Returns:
        Spotify: An authenticated Spotify client instance.
    """

    # Attempt to authenticate with Spotify using the spotify_login function.
    try:
        auth_manager = spotify_login()
        logging.info("Spotify authentication successful")
    except Exception as e:
        logging.error(f"Error during Spotify authentication: {e}")
        raise

    # Create a new Spotify client instance with the provided authentication manager.
    sp = Spotify(auth_manager=auth_manager)
    logging.info("Spotify client instance created")

    # Return the authenticated Spotify client instance.
    return sp


def extract_uri(tracks: List[Dict]) -> List[str]:
    """
    Extracts the URIs of favorite songs from the given list of tracks.

    Args:
        tracks: List of track dictionaries.

    Returns:
        List of song URIs.
    """
    # Extract song URIs
    songs_uris = [f['track']['uri'] for f in tracks]
    
    # Log the number of extracted song URIs
    logging.info(f"Extracted {len(songs_uris)} favorite song URIs")
    
    return songs_uris


def spotify_join_names(spotify_track):
    res = f"{spotify_track['track']['name']} {' '.join([a['name'] for a in spotify_track['track']['artists']])}"
    return res


@click.command()
@click.option('--order', '-o', default='similarity', type=click.Choice(['similarity', 'random']), help='Specify the order for mixing the playlist. Supported values: similarity, random.')
@click.option('--playlist_name', '-p', required=False, help='Name of the playlist to be processed.')
def main(
    order: str,
    playlist_name: str
    ):
    """
    Main function.

    Args:
        order: The order for mixing the playlist.
        playlist_name: Name of the playlist to be processed.

    Returns:
        None

    This function initializes the Spotify API client, retrieves the user's favorite songs,
    mixes the playlist with the given order, and outputs the result or handles it as needed.
    """
    # Initialize the Spotify API client
    spotipy_client = spotify_api()

    if playlist_name:

        c_user = spotipy_client.current_user()
        
        # Fetch the current user's playlists
        playlists = fetch_playlists(
            spotipy_client=spotipy_client,
            user_id=c_user['id']
        )

        fuzzy_match = process.extractOne(playlist_name, [p['name'] for p in playlists])
        playlist_matched = fuzzy_match[0]
        for playlist in playlists:
            if playlist['name'] == playlist_matched:
                break
        
        songs = fetch_playlist_songs(
            spotipy_client=spotipy_client,
            playlist=playlist
        )
    else:

        # Retrieve the user's favorite songs
        # starred_songs = user_favorites(spotipy_client=spotipy_client)
        playlist_matched = "Starred"
        songs = fetch_starred_songs(spotipy_client=spotipy_client)
    tracks_uris = extract_uri(songs)

    # Mix the playlist with the given order
    spotify_songs_uris = playlist_mixer(
        sp=spotipy_client,
        tracks_uris=tracks_uris,
        order=order
    )

    # Initialize the time variable with the current datetime to use in creating playlist names.
    time = datetime.now()
    tm = time.strftime("%Y-%m-%d %H:%M")
    # Format the timestamp for including in the playlist name.
    playlist_new_name = f'SpotiShuffle - {playlist_matched} - {tm}'
    logging.info("Pushing the playlist '%s' to Spotify", playlist_new_name)
    create_playlist(
        spotipy_client=spotipy_client,
        playlist_name=playlist_new_name,
        spotify_songs_uris=spotify_songs_uris
    )

    # Log the successful completion of playlist update.
    logging.info("Playlist '%s' processed successfully", playlist_matched)


if __name__ == '__main__':
    main()
