import mongoose from "mongoose";

const chatSchema=mongoose.Schema({
    userId:{type:String,trim:true},
    question:{type:String,trim:true},
    answer:{type:String,trim:true}
})

const Chat_db = mongoose.model('Chat', chatSchema);

export default Chat_db;
