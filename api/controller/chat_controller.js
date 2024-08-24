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

    static getInitalChats=async(req,res)=>{
        try {
            const firstDocs = await Chat_db.aggregate([
                // Sort by chatId and any other criteria (e.g., by creation time)
                { $sort: { chatId: 1, _id: 1 } },
                
                // Group by chatId and pick the first document in each group
                {
                    $group: {
                        _id: "$chatId",
                        firstDoc: { $first: "$$ROOT" }
                    }
                },
    
                // Optionally, replace the root document with the firstDoc for easier access
                {
                    $replaceRoot: {
                        newRoot: "$firstDoc"
                    }
                }
            ]);
    
            return res.json(firstDocs);


        } catch (error) {
            console.log(error)
        }
    }
}

export default chatCotroller