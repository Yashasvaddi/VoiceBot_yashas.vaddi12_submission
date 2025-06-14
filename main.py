import streamlit as st
import pandas as pd
import io
from modules.response_gen import generate_response

st.title("LenDen Mitra Query System")

# Single query input
st.subheader("Single Query")
query = st.text_input("Enter your question:")
if st.button("Get Response", key="single_query"):
    if query:
        with st.spinner("Generating response..."):
            response, source, confidence = generate_response(query)
            st.write(f"**Response:** {response}")
            st.write(f"**Source:** {source} (Confidence: {confidence}%)")
    else:
        st.warning("Please enter a question")

# CSV file upload
st.subheader("Batch Processing")
uploaded_file = st.file_uploader("Upload CSV file with questions", type="csv")
if uploaded_file is not None:
    # Read the CSV
    df = pd.read_csv(uploaded_file)
    
    # Determine question column name
    question_col = next((col for col in df.columns if col.lower() in ["question", "questions"]), None)
    
    if question_col:
        if st.button("Process CSV"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process each question
            responses = []
            sources = []
            confidences = []
            
            for i, question in enumerate(df[question_col]):
                status_text.text(f"Processing question {i+1}/{len(df)}")
                response, source, confidence = generate_response(question)
                responses.append(response)
                sources.append(source)
                confidences.append(confidence)
                progress_bar.progress((i + 1) / len(df))
            
            # Add results to dataframe
            df["Model_Response"] = responses
            df["Response_Source"] = sources
            df["Confidence"] = confidences
            
            # Display results
            st.success("Processing complete!")
            st.dataframe(df)
            
            # Download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Results",
                csv,
                "results_with_responses.csv",
                "text/csv",
                key="download-csv"
            )
    else:
        st.error("CSV file must contain a column named 'Question' or 'Questions'")