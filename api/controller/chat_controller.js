import Chat_db from "../models/chat_model.js"
import axios from "axios"

class chatCotroller
{
    static saveChat=async(req,res)=>{
        try {
            const {question,email,chatId}=req.body
            console.log("data:",question,email,chatId)
            const answer = await axios.post('https://sih-fwt9.onrender.com/chat', {
                message: question
              });
            

            console.log(answer.data)  

            const chat = new Chat_db({
                email,
                chatId,
                question,
                answer:answer.data.response,
              });
          
              // Save chat to MongoDB
              await chat.save();

              return res.json(answer.data)

            
        } catch (error) {
            console.log(error)
        }
    }

    static getChat=async(req,res)=>{
        try {
            
        } catch (error) {
            console.log(error)
        }
    }
}

export default chatCotroller