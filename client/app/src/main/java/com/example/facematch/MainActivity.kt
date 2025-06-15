package com.example.facematch

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.AppCompatImageView
import androidx.core.content.ContextCompat
import com.daimajia.androidanimations.library.Techniques
import com.daimajia.androidanimations.library.YoYo
import com.github.dhaval2404.imagepicker.ImagePicker
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import com.google.firebase.firestore.ListenerRegistration
import com.google.firebase.firestore.ktx.firestore
import com.google.firebase.ktx.Firebase
import io.github.jan.supabase.storage.storage
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await
import java.io.InputStream

class MainActivity : AppCompatActivity() {

    // UI элементы
    private lateinit var previewImage: AppCompatImageView
    private lateinit var uploadCard: MaterialCardView
    private lateinit var progressBar: ProgressBar
    private lateinit var resultContainer: View
    private lateinit var uploadButton: MaterialButton
    private lateinit var resultsListContainer: LinearLayout

    // Данные приложения
    private var firestoreListener: ListenerRegistration? = null
    private var curPhotoId: String = ""
    private var currentImageUri: Uri? = null

    private val imagePickerLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            val uri: Uri? = result.data?.data
            uri?.let {
                currentImageUri = it
                previewImage.setImageURI(it)
                uploadImage(it)
            }
        } else if (result.resultCode == ImagePicker.RESULT_ERROR) {
            Toast.makeText(this, "Ошибка: ${ImagePicker.getError(result.data)}", Toast.LENGTH_SHORT)
                .show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Инициализация UI элементов
        previewImage = findViewById(R.id.previewImage)
        uploadCard = findViewById(R.id.uploadCard)
        progressBar = findViewById(R.id.progressBar)
        resultContainer = findViewById(R.id.resultContainer)
        uploadButton = findViewById(R.id.uploadButton)
        resultsListContainer = findViewById(R.id.resultsListContainer)

        // Начальное состояние UI
        resetUIState()

        // Обработчик кнопки загрузки
        uploadButton.setOnClickListener {
            ImagePicker.with(this)
                .crop()
                .compress(1024)
                .createIntent { intent ->
                    imagePickerLauncher.launch(intent)
                }
        }
    }

    private fun resetUIState() {
        previewImage.visibility = View.GONE
        uploadCard.visibility = View.VISIBLE
        progressBar.visibility = View.GONE
        resultContainer.visibility = View.GONE
        uploadButton.isEnabled = true
    }

    private fun showLoadingState() {
        uploadCard.visibility = View.GONE
        previewImage.visibility = View.VISIBLE
        progressBar.visibility = View.VISIBLE
        resultContainer.visibility = View.GONE
        uploadButton.isEnabled = false
    }

    private fun uploadImage(uri: Uri) {
        showLoadingState()

        CoroutineScope(Dispatchers.IO).launch {
            try {
                // Чтение файла
                val inputStream: InputStream? = contentResolver.openInputStream(uri)
                val bytes = inputStream?.readBytes() ?: throw Exception("Ошибка чтения файла")
                val name = System.currentTimeMillis().toString()

                // Загрузка в Supabase Storage
                Supabase.client.storage["user-photo"].upload(
                    path = "uploads/${name}.jpg",
                    data = bytes,
                    upsert = false
                )

                // Создание документа в Firestore
                Firebase.firestore.runTransaction { transaction ->
                    val docRef = Firebase.firestore.collection("people").document(name)
                    if (transaction.get(docRef).exists()) {
                        throw Exception("Документ уже существует")
                    }
                    transaction.set(docRef, hashMapOf<String, Any>())
                }.await()

                curPhotoId = name
                setupFirestoreListener()

                runOnUiThread {
                    Toast.makeText(
                        this@MainActivity,
                        "Фото загружено",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            } catch (e: Exception) {
                runOnUiThread {
                    Toast.makeText(
                        this@MainActivity,
                        "Ошибка: ${e.message}",
                        Toast.LENGTH_LONG
                    ).show()
                    resetUIState()
                }
            }
        }
    }

    private fun setupFirestoreListener() {
        firestoreListener?.remove()
        if (curPhotoId.isBlank()) return

        val docRef = Firebase.firestore.collection("people").document(curPhotoId)

        firestoreListener = docRef.addSnapshotListener { snapshot, error ->
            if (error != null) {
                runOnUiThread {
                    Toast.makeText(this, "Ошибка: ${error.message}", Toast.LENGTH_SHORT).show()
                    resetUIState()
                }
                return@addSnapshotListener
            }

            snapshot?.let { doc ->
                if (doc.exists() && currentImageUri != null) {
                    val resultsList = doc.get("result") as? List<Map<String, Any>> ?: emptyList()

                    if (resultsList.isNotEmpty()) {
                        // Собираем все результаты
                        var faceResults = mutableListOf<FaceResult>()
                        for (result in resultsList) {
                            val labels = result["labels"] as? List<String> ?: listOf("Нет лица")
                            if (labels[0] == "Нет лица") {
                                faceResults = mutableListOf()
                                break
                            }
                            val score = (result["score"] as? Double) ?: 0.0
                            val pos = result["pos"] as? List<Int> ?: listOf(0, 0, 0, 0)

                            faceResults.add(FaceResult(labels, score, pos))
                        }

                        runOnUiThread {
                            showResults(currentImageUri, faceResults)
                            currentImageUri = null

                            Firebase.firestore.runTransaction { transaction ->
                                if (transaction.get(docRef).exists()) {
                                    transaction.delete(docRef)
                                }
                            }
                        }
                    } else {
                        runOnUiThread {
                            Toast.makeText(this, "Обработка...", Toast.LENGTH_SHORT).show()
                        }
                    }
                }
            }
        }
    }

    data class FaceResult(
        val names: List<String>,
        val confidence: Double,
        val boundingBox: List<Int> // [x1, y1, x2, y2]
    )

    private fun showResults(imageUri: Uri?, results: List<FaceResult>) {
        resultsListContainer.removeAllViews()
        previewImage.setImageURI(imageUri)

        if (results.isEmpty()) {
            val emptyView = TextView(this).apply {
                text = "Лица не обнаружены"
                setTextColor(ContextCompat.getColor(this@MainActivity, R.color.text_secondary))
                gravity = Gravity.CENTER
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                )
            }
            resultsListContainer.addView(emptyView)
        } else {
            // Загружаем оригинальное изображение для обрезки
            val inputStream = contentResolver.openInputStream(imageUri!!)
            val originalBitmap = BitmapFactory.decodeStream(inputStream)
            inputStream?.close()

            results.forEach { result ->
                val card = layoutInflater.inflate(R.layout.item_face_result, resultsListContainer, false).apply {
                    findViewById<TextView>(R.id.faceName).text = result.names.joinToString(", ")
                    findViewById<TextView>(R.id.faceConfidence).text = "Точность: ${"%.2f".format(result.confidence)}%"

                    // Обрезаем изображение по координатам лица
                    val (x1, y1, x2, y2) = result.boundingBox
                    if (x1 >= 0 && y1 >= 0 && x2 <= originalBitmap.width && y2 <= originalBitmap.height) {
                        val croppedBitmap = Bitmap.createBitmap(
                            originalBitmap,
                            x1, y1,
                            x2 - x1, y2 - y1
                        )
                        findViewById<ImageView>(R.id.faceImage).setImageBitmap(croppedBitmap)
                    } else {
                        findViewById<ImageView>(R.id.faceImage).setImageResource(R.drawable.placeholder_image)
                    }
                }
                resultsListContainer.addView(card)
            }
        }

        resultContainer.visibility = View.VISIBLE
        progressBar.visibility = View.GONE
        uploadButton.isEnabled = true
    }
}