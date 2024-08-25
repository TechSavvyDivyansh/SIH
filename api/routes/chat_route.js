import express from 'express'
import chatCotroller from '../controller/chat_controller.js'

const router=express.Router()

router.post('/savechat',chatCotroller.saveChat)
router.post('/get-initial-chat',chatCotroller.getInitalChats)
router.post('/get-chat-messages',chatCotroller.getSpecificChatMessages)
router.get('/get-all-chats',chatCotroller.getAllChat)

export default router