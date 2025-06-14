import google.generativeai as genai


def translate():
    genai.configure(api_key="AIzaSyC3vNkSnEJl-eFloSm9M4Bw0F_cJv2vusY")
    model = genai.GenerativeModel("gemini-2.0-flash")


    text=model.generate_content(f"Is the given audio/text sample {ans} in hindi or english. Reply 1 if hindi and 0 for english")

    val=text.text

    final=int(val)

    print(final)