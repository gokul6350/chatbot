
import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.write("# Welcome to PDF LM 👋")

    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        # PDF LM: PDF Language Model 🚀

**PDF LM** stands for **PDF Language Model**. It is designed to offer a comprehensive suite of tools for managing and interpreting PDF documents. The key functionalities of PDF LM include:

1. **Summarize PDF** 📝: Quickly generate concise summaries of lengthy PDF documents, making it easier to grasp the essential information without going through the entire content.

2. **Annotate Important Points** 📌: Effortlessly highlight and annotate key points within your PDF files, ensuring that critical information is easily accessible and well-documented.

3. **Visualize Data** 📊: Transform complex data from PDFs into clear, understandable charts and visual representations, facilitating better analysis and decision-making.

## Integration with Gemini API 🔗

This tool leverages the power of the **Gemini API** to enhance its functionality. To integrate the Gemini API, ensure that you add your API key to the `.env` file in your project:

```plaintext
GEMINI_API_KEY=your_api_key_here
```
    """
    )


if __name__ == "__main__":
    run()


