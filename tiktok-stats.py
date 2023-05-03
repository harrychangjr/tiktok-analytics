from TikTokApi import TikTokApi
import pprint

def main():
    verify_fp = "verify_lh70h7bq_KVGJUp1A_3X29_4G4Z_83m8_HoTlQwPRxQNZ"
    api = TikTokApi(custom_verify_fp=verify_fp)

    user = api.user(username="therock")

    for video in user.videos():
        print(video.id)

    for liked_video in api.user(username="public_likes").videos():
        print(liked_video.id)
    # Initialize the TikTokApi instance
    #with TikTokApi() as api:
        #user = api.user(username='therock')
        #user.as_dict # -> dict of the user_object
        #for video in user.videos():
            #video.as_dict
        #print(video)

    # Replace 'your_username' with your TikTok username
        #username = 'questhireahero'

    # Get user information
        #user_info = api.get_user(username)
    
    # Extract user ID
        #user_id = user_info['userInfo']['user']['id']
    
    # Get user's videos
        #user_videos = api.by_username(username, count=100)  # Count can be set to any number up to 100

    # Calculate total views
        #total_views = 0
        #for video in user_videos:
            #total_views += video['stats']['playCount']

    # Print the results
        #pprint.pprint(user_info['userInfo']['stats'])
        #print(f"Total views: {total_views}")

if __name__ == "__main__":
    main()

