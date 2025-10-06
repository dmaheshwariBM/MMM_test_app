import os
from core import remote_store as rs


def main():
    if not os.getenv("MONGO_URI"):
        print("MONGO_URI not set. Set MONGO_URI in a .env file or environment to run smoke test.")
        return

    # upload README.txt as a small smoke test
    local = os.path.join(os.path.dirname(__file__), "..", "README.txt")
    local = os.path.abspath(local)
    print("Uploading:", local)
    file_id = rs.upload_file(None, "smoke_README.txt", local, metadata={"source": "smoke-test"})
    print("Uploaded file id:", file_id)

    files = rs.list_files(None, prefix="smoke_")
    print("Files with prefix 'smoke_':", files)

    # download back
    out = os.path.join(os.path.dirname(__file__), "tmp_smoke_readme.txt")
    rs.download_file(None, file_id, out)
    print("Downloaded to:", out)

    # config test
    cfg = {"hello": "world"}
    saved = rs.save_config("smoke_test_config", cfg)
    print("Saved config:", saved)


if __name__ == "__main__":
    main()
