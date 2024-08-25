import Chat_db from "../models/chat_model.js"
import axios from "axios"

class chatCotroller
{
    static saveChat=async(req,res)=>{
        try {
            const {question,email,chatId}=req.body
            console.log("data:",question,email,chatId)
            const answer = await axios.post('https://atharvmendhe18-sit-internal.hf.space/process', {
                text: question
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
            const { email } = req.body; // Or req.body if email is sent in the body

        if (!email) {
            return res.status(400).json({ error: 'Email parameter is required' });
        }

        const firstDocs = await Chat_db.aggregate([
            {
                $match: { email: email }
            },

            { $sort: { chatId: 1, _id: 1 } },

            {
                $group: {
                    _id: "$chatId",
                    firstDoc: { $first: "$$ROOT" }
                }
            },

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


    static getSpecificChatMessages=async(req,res)=>{
        try {

            const {chatId,email}=req.body
            const chats=await Chat_db.find({
                chatId:chatId,
                email:email 
            })

            return res.json(chats)
            
        } catch (error) {
            console.log(error)
        }
    }


    static getAllChat=async(req,res)=>{
        try {

            const chats=await Chat_db.find({})
            return res.json(chats)
            
        } catch (error) {
            console.log(error)
        }
    }


}

export default chatCotroller