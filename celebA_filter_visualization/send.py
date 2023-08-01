from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

# 用你的JSON密钥文件路径替换以下路径
KEY_FILE_PATH = '/home/clydechx/FairFace/celebA_filter_visualization/client_secret.json'

# 使用服务帐户的凭据
credentials = Credentials.from_service_account_file(
    KEY_FILE_PATH, scopes=['https://www.googleapis.com/auth/drive'])

# 创建Drive API客户端
drive_service = build('drive', 'v3', credentials=credentials)

# 创建文件元数据
file_metadata = {
    'name': 'LFW+_results.zip',
    'mimeType': '*/*'
}

# 创建媒体文件上传对象
media = MediaFileUpload(
    '/home/clydechx/FairFace/LFW+_filter_visualization/lfw+_results.zip', mimetype='*/*', resumable=True)

# 创建文件
file = drive_service.files().create(
    body=file_metadata, media_body=media, fields='id').execute()

file_id = file.get('id')
print('File ID:', file_id)

# 将特定的电子邮件地址添加为文件共享者
email_address = 'clydechx@umich.edu'  # 用你的电子邮件地址替换


def share_with_email(service, file_id, email_address):
    try:
        permission = {
            'type': 'user',
            'role': 'reader',  # 或 'writer' 以允许编辑
            'emailAddress': email_address
        }
        command = service.permissions().create(
            fileId=file_id,
            body=permission,
            fields='id'
        )
        command.execute()
        print(f'Shared with {email_address}')
    except HttpError as error:
        print(f'An error occurred: {error}')


share_with_email(drive_service, file_id, email_address)
