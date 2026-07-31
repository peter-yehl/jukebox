import webbrowser

current_album = None

def search_youtube(query):
    request = youtube.search().list(
        q=query,
        part="snippet",
        maxResults=1,
        type="video"
    )

    response = request.execute()

    if not response["items"]:
        return None

    return response["items"][0]["id"]["videoId"]


def build_query(album):
    artist = album["artist"]
    title = album["album"]
    vibes = " ".join(album["vibes"])

    # First try to find an album visualizer
    return f"{artist} {title} visualizer {vibes}"


def play_lyric(recognized):
    global current_album

    # Don't reopen the same video repeatedly
    if recognized == current_album:
        return

    current_album = recognized

    album = next(
        (
            a for a in metadata
            if f"{a['artist']} {a['album']}" == recognized
        ),
        None
    )

    if album is None:
        print("Album not found.")
        return

    query = build_query(album)
    print("Searching:", query)

    video = search_youtube(query)

    if video:
        webbrowser.open(f"https://www.youtube.com/watch?v={video}")