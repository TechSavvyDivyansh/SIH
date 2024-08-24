import express from 'express'
import mongoose from 'mongoose'
import dotenv from 'dotenv'
import cors from 'cors'
import ChatRouter from '../api/routes/chat_route.js'


dotenv.config()

mongoose.connect(process.env.MONGO).then(()=>{
    console.log("mongodb connected")
}).catch((err)=>{
    console.log(err)
})


const app=express()

app.use(express.json())
app.use(cors());

app.listen('https://doj-backend.onrender.com',()=>{
    console.log("server running on port 4000!!")
})

app.use('/api',ChatRouter)