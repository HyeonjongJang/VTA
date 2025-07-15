import streamlit as st

def first_page():
    st.image("./asset/kyunghee.png", width=300)
    st.title("Kyung Hee Regulations Virtual Assistant")

    st.markdown("""
    ### About this Virtual Assistant
    This Virtual Assistant provides answers based on the updated datasets of regulations, internal rules, and guidelines from Kyung Hee University's Regulation Management System.
    Reference : https://rule.khu.ac.kr/lmxsrv/main/main.do
    
    **Important Notice:**
    - This tool is exclusively for the Kyung Hee University's Regulations Search. **Do not** use it for any other purposes.
    - There is a rate limit on GPT-4 usage. Please be mindful of your usage to ensure that all students have an equal opportunity to benefit from this tool.
    - **Student IDs found to be using this tool for purposes other than for the Regulations Search, or with abnormally high usage, may have their access revoked.**
    - Conversations with the Virtual Assistant will be stored and can be used for research purposes. However, your student ID will be thoroughly anonymized. **Do not** include any identifying information in your conversations.
    - Since the model may hallucinate, for matters directly related to grades (e.g., project submission deadlines), be sure to check the relevant documents directly or contact the Assistant.
    - By using this Virtual Assistant, you agree to these terms and conditions.
    """)

    # Contact Info
    st.markdown("""
    **Contact Info:**
    - If you have any questions or need assistance, please contact: lezelamu@naver.com
    - This program was developed by [KYUNGHEE AIMSlab](https://sites.google.com/khu.ac.kr/aims).
    """)

    # Agreement Checkbox
    agreement = st.checkbox("I agree to the terms and conditions stated above.")
    
    student_id = st.text_input("Submit your Student ID to get started!")

    if st.button("Submit"):
        if agreement:
            if student_id in st.secrets["student_ids"]:
                st.session_state["student_id"] = student_id
                st.rerun()
            else:
                st.error("Invalid Student ID. Please try again.")
        else:
            st.error("You must agree to the terms and conditions before proceeding.")


