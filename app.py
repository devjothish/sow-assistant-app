import streamlit as st
from sow_assistant import SOWAssistant

# Initialize the SOW Assistant
@st.cache_resource
def initialize_assistant():
    return SOWAssistant()

assistant = initialize_assistant()

st.title("SOW Assistant Chatbot")

# Input for generating SOW
st.header("Generate SOW")
sow_input = st.text_area("Enter project details for SOW generation:", height=200)
if st.button("Generate SOW"):
    if sow_input:
        with st.spinner("Generating SOW..."):
            result = assistant.generate_sow(sow_input)
        st.subheader("Generated SOW:")
        st.text_area("SOW Content", result["sow"], height=400)
        
        st.subheader("Reference Sources:")
        for source in result["sources"]:
            st.write(f"Source: {source['source']}")
            st.write(f"Similarity: {source['similarity']}")
            st.text(source['content'])
            st.markdown("---")
    else:
        st.warning("Please enter project details to generate SOW.")

# Input for customizing SOW
st.header("Customize SOW")
customize_input = st.text_area("Enter customization request:", height=100)
if st.button("Customize SOW"):
    if customize_input:
        with st.spinner("Customizing SOW..."):
            result = assistant.customize_sow(customize_input)
        st.subheader("Customized SOW:")
        st.text_area("Updated SOW Content", result["sow"], height=400)
        
        st.subheader("Reference Sources for Customization:")
        for source in result["sources"]:
            st.write(f"Source: {source['source']}")
            st.write(f"Similarity: {source['similarity']}")
            st.text(source['content'])
            st.markdown("---")
    else:
        st.warning("Please enter a customization request.")

# Export SOW to DOCX
st.header("Export SOW to DOCX")
export_prompt = st.text_input("Enter any specific export instructions (optional):")
if st.button("Export SOW"):
    with st.spinner("Exporting SOW to DOCX..."):
        result = assistant.export_sow_docx(export_prompt)
    st.success(result)
    st.download_button(
        label="Download SOW DOCX",
        data=open("generated_sow.docx", "rb").read(),
        file_name="generated_sow.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )